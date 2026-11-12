// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    ClusterNode,
    NodeNameResult,
    RemoteManagementCapability
} from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { NodeNameModal } from '@/ui/components/NodeName/NodeNameModal'
import { ServiceCallError } from '@/ui/api/service-transport'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'

const setNodeName = vi.fn<IClusterApi['setNodeName']>()
const setPeerName = vi.fn<IClusterApi['setPeerName']>()
const setPeerNickname = vi.fn<IClusterApi['setPeerNickname']>()
const getRemoteManagementForNode = vi.fn<IClusterApi['getRemoteManagementForNode']>()
const onClose = vi.fn<() => void>()
const DENIED_MESSAGE =
    'Remote management is blocked on this device. Change permission locally on that device.'

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
                target={{ kind: 'self', nodeUuid: 'self-uuid', shownName, customName }}
                open
                onClose={onClose}
            />
        )
    )
}

async function renderPeerModal(node: ClusterNode = peer): Promise<void> {
    await act(async () =>
        root.render(<NodeNameModal target={{ kind: 'peer', node }} open onClose={onClose} />)
    )
}

function capability(state: RemoteManagementCapability['state']): RemoteManagementCapability {
    return { version: 1, allowed: state === 'allowed', revision: '1', state }
}

function visibleText(): string {
    return document.body.textContent ?? ''
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    setNodeName.mockReset()
    setPeerName.mockReset()
    setPeerNickname.mockReset()
    getRemoteManagementForNode.mockReset()
    onClose.mockReset()
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: {
            cluster: { setNodeName, setPeerName, setPeerNickname, getRemoteManagementForNode }
        }
    })
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { platform: 'Windows' }
    })
    useConnectionStore.setState({ selfId: 'self-uuid' })
    useClusterInvitationsStore.setState({
        members: [self, peer],
        nameChange: null,
        peerNameChanges: {}
    })
    useNodeConfigurationStore.setState({ management: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useClusterInvitationsStore.setState({ members: [], nameChange: null, peerNameChanges: {} })
    useNodeConfigurationStore.setState({ management: new Map() })
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
            new ServiceCallError('node name is longer than 128 bytes', -32602)
        )
        await renderModal(null)
        await typeName('x'.repeat(200))
        await click('Save')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe(
            'node name is longer than 128 bytes'
        )
        expect(onClose).not.toHaveBeenCalled()
    })
})

describe('node name modal for a peer', () => {
    it('offers Save, Set nickname and Use default name when the peer allows management', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        await renderPeerModal()
        expect(getRemoteManagementForNode).toHaveBeenCalledExactlyOnceWith('peer-uuid')
        expect(button('Save')).toBeDefined()
        expect(button('Set nickname')).toBeDefined()
        expect(button('Use default name')).toBeDefined()
        expect(visibleText()).toContain(
            'Renames Studio on that device. Every node in the cluster will show the new name.'
        )
        expect(visibleText()).toContain(
            "A nickname is only used on this device. Other devices keep showing the node's own name."
        )
        expect(visibleText()).not.toContain('has not allowed this device')
    })

    it.each<RemoteManagementCapability['state']>([
        'denied',
        'unreviewed',
        'unavailable',
        'unsupported'
    ])('offers only Set nickname when the state is %s', async state => {
        getRemoteManagementForNode.mockResolvedValue(capability(state))
        await renderPeerModal()
        expect(button('Save')).toBeUndefined()
        expect(button('Use default name')).toBeUndefined()
        expect(button('Set nickname')).toBeDefined()
        expect(visibleText()).toContain(
            "Studio has not allowed this device to manage it, so you can't rename it from here. You can still give it a nickname."
        )
    })

    it('hides Save while the permission read is pending', async () => {
        getRemoteManagementForNode.mockReturnValue(new Promise(() => {}))
        await renderPeerModal()
        expect(button('Save')).toBeUndefined()
        expect(button('Set nickname')).toBeDefined()
        expect(visibleText()).not.toContain('has not allowed this device')
    })

    it('prefills the peer name, or its nickname when one is set', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        await renderPeerModal()
        expect(nameInput().value).toBe('Studio')
        await act(async () => root.render(null))
        await renderPeerModal({ ...peer, nickname: 'Kitchen', displayName: 'Kitchen' })
        expect(nameInput().value).toBe('Kitchen')
    })

    it('renames the peer with the trimmed name and closes', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        setPeerName.mockResolvedValue({ nodeUuid: 'peer-uuid', name: 'Den', customName: 'Den' })
        await renderPeerModal()
        await typeName('  Den  ')
        await click('Save')
        expect(setPeerName).toHaveBeenCalledExactlyOnceWith('peer-uuid', 'Den')
        expect(setNodeName).not.toHaveBeenCalled()
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('sends null for the peer from Use default name', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        setPeerName.mockResolvedValue({
            nodeUuid: 'peer-uuid',
            name: 'PAIR 9f8e7d6c',
            customName: null
        })
        await renderPeerModal()
        await click('Use default name')
        expect(setPeerName).toHaveBeenCalledExactlyOnceWith('peer-uuid', null)
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('sets the trimmed nickname without the grant and closes', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('denied'))
        setPeerNickname.mockResolvedValue({
            nodeUuid: 'peer-uuid',
            nickname: 'Kitchen',
            displayName: 'Kitchen'
        })
        await renderPeerModal()
        await typeName('  Kitchen  ')
        await click('Set nickname')
        expect(setPeerNickname).toHaveBeenCalledExactlyOnceWith('peer-uuid', 'Kitchen')
        expect(setPeerName).not.toHaveBeenCalled()
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('clears the nickname only when one is set', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('denied'))
        await renderPeerModal()
        expect(button('Clear nickname')).toBeUndefined()
        await act(async () => root.render(null))
        setPeerNickname.mockResolvedValue({
            nodeUuid: 'peer-uuid',
            nickname: null,
            displayName: 'Studio'
        })
        await renderPeerModal({ ...peer, nickname: 'Kitchen', displayName: 'Kitchen' })
        await click('Clear nickname')
        expect(setPeerNickname).toHaveBeenCalledExactlyOnceWith('peer-uuid', null)
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('shows the target message and switches to Nickname-only after a denial', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        setPeerName.mockRejectedValue(new ServiceCallError(DENIED_MESSAGE, -32012))
        await renderPeerModal()
        await click('Save')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe(DENIED_MESSAGE)
        expect(onClose).not.toHaveBeenCalled()
        expect(button('Save')).toBeUndefined()
        expect(button('Use default name')).toBeUndefined()
        expect(button('Set nickname')).toBeDefined()
    })

    it('keeps Save hidden after a denial when a later nickname save fails', async () => {
        getRemoteManagementForNode
            .mockResolvedValueOnce(capability('allowed'))
            .mockResolvedValue(capability('denied'))
        setPeerName.mockRejectedValue(new ServiceCallError(DENIED_MESSAGE, -32012))
        setPeerNickname.mockRejectedValue(new ServiceCallError('device is offline', -32000))
        await renderPeerModal()
        await click('Save')
        expect(getRemoteManagementForNode).toHaveBeenCalledTimes(2)
        await click('Set nickname')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe('device is offline')
        expect(button('Save')).toBeUndefined()
        expect(button('Use default name')).toBeUndefined()
    })

    it('shows a validation message from the target and keeps Save', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        setPeerName.mockRejectedValue(
            new ServiceCallError('node name is longer than 128 bytes', -32602)
        )
        await renderPeerModal()
        await typeName('x'.repeat(200))
        await click('Save')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe(
            'node name is longer than 128 bytes'
        )
        expect(button('Save')).toBeDefined()
    })

    it('warns about a duplicate shown name without blocking Save or Set nickname', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        await renderPeerModal()
        await typeName('pair 1a2b3c4d')
        expect(document.querySelector('[role="status"]')?.textContent).toBe(
            'Another node is already shown as "PAIR 1a2b3c4d". You can still save; PAIR adds a hardware label to tell them apart.'
        )
        expect(button('Save')?.disabled).toBe(false)
        expect(button('Set nickname')?.disabled).toBe(false)
    })
})

describe('node name modal across opens', () => {
    function pendingSave() {
        let resolve: (value: NodeNameResult) => void = () => {
            throw new Error('save not started')
        }
        const promise = new Promise<NodeNameResult>(finish => {
            resolve = finish
        })
        return { promise, resolve }
    }

    async function reopen(): Promise<void> {
        await act(async () => root.render(null))
        await renderModal(null)
    }

    it('keeps a save from an earlier open in flight when reopened', async () => {
        const save = pendingSave()
        setNodeName.mockReturnValue(save.promise)
        await renderModal(null)
        await click('Save')
        await reopen()
        expect(useClusterInvitationsStore.getState().nameChange).toEqual({ phase: 'saving' })
        expect(nameInput().disabled).toBe(true)
        expect(button('Save')?.disabled).toBe(true)
        expect(setNodeName).toHaveBeenCalledOnce()
    })

    it('does not close a later open when an earlier save finishes', async () => {
        const save = pendingSave()
        setNodeName.mockReturnValue(save.promise)
        await renderModal(null)
        await click('Save')
        await reopen()
        await act(async () =>
            save.resolve({ nodeUuid: 'self-uuid', name: 'Desk', customName: 'Desk' })
        )
        expect(onClose).not.toHaveBeenCalled()
        expect(button('Save')?.disabled).toBe(false)
    })

    it('does not show an error from an earlier open', async () => {
        useClusterInvitationsStore.setState({ nameChange: { phase: 'error', message: 'old' } })
        await renderModal(null)
        expect(document.querySelector('[role="alert"]')).toBeNull()
    })

    it('shows an error from a save made in this open', async () => {
        useClusterInvitationsStore.setState({ nameChange: { phase: 'error', message: 'old' } })
        setNodeName.mockRejectedValue(new ServiceCallError('new problem', -32602))
        await renderModal(null)
        await click('Save')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe('new problem')
    })

    it('offers Save to a peer again after a denial from an earlier open', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        useClusterInvitationsStore.setState({
            peerNameChanges: {
                'peer-uuid': { phase: 'error', message: DENIED_MESSAGE, denied: true }
            }
        })
        await renderPeerModal()
        expect(button('Save')).toBeDefined()
        expect(document.querySelector('[role="alert"]')).toBeNull()
    })

    it('keeps the in-flight guard of this node when a peer save succeeds', async () => {
        getRemoteManagementForNode.mockResolvedValue(capability('allowed'))
        setPeerName.mockResolvedValue({ nodeUuid: 'peer-uuid', name: 'Den', customName: 'Den' })
        await renderPeerModal()
        useClusterInvitationsStore.setState({ nameChange: { phase: 'saving' } })
        await click('Save')
        expect(onClose).toHaveBeenCalledOnce()
        expect(useClusterInvitationsStore.getState().nameChange).toEqual({ phase: 'saving' })
        expect(useClusterInvitationsStore.getState().peerNameChanges['peer-uuid']).toBeUndefined()
    })

    it('keeps a peer outcome when a save of this node succeeds', async () => {
        setNodeName.mockResolvedValue({ nodeUuid: 'self-uuid', name: 'Desk', customName: 'Desk' })
        useClusterInvitationsStore.setState({ peerNameChanges: { other: { phase: 'saving' } } })
        await renderModal(null)
        await click('Save')
        expect(onClose).toHaveBeenCalledOnce()
        expect(useClusterInvitationsStore.getState().peerNameChanges.other).toEqual({
            phase: 'saving'
        })
    })
})
