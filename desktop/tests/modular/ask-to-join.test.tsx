// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    AvailableNode,
    Invite,
    JoinRequest,
    JoinRequestsSnapshot
} from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { AddNodeModal } from '@/ui/components/AddNodeModal'
import AvailableNodeCard from '@/ui/components/ClusterSettings/AvailableNode'
import ClusterSettings from '@/ui/components/ClusterSettings/ClusterSettings'
import JoinRequestCard from '@/ui/components/ClusterSettings/JoinRequestCard'
import { InviteApprovalModal } from '@/ui/components/InviteApprovalModal'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { joinRequestMessage } from '@/ui/utils/cluster-invite-error'

const peer: AvailableNode = {
    id: 'peer',
    name: 'Member',
    ipAddress: '192.0.2.1',
    port: 14321,
    lastSeen: 1,
    trusted: false,
    clustered: true
}
const request: JoinRequest = {
    requestId: 'jr-one',
    direction: 'outgoing',
    peerNodeUuid: 'peer',
    peerName: 'Member',
    peerAddress: '192.0.2.1',
    peerPort: 14321,
    state: 'awaiting-approval',
    inviteId: null,
    reason: '',
    createdAt: 1,
    expiresAt: 60000
}
const baseline: JoinRequestsSnapshot = {
    supported: true,
    canAskToJoin: true,
    requiresSoloReplacement: false,
    blockedReason: '',
    requests: []
}
const invite: Invite = {
    inviteId: 'child',
    fromNodeId: 'member',
    fromNodeUuid: 'peer',
    fromNodeName: 'Member',
    toNodeId: null,
    clusterId: 'cluster',
    clusterFriendlyName: 'Home',
    pin: '123456',
    state: 'pending',
    reason: '',
    createdAt: 1,
    respondedAt: null
}
const ask = vi.fn<IClusterApi['askToJoin']>()
const inviteNode = vi.fn<IClusterApi['inviteNode']>()
const status = vi.fn<IClusterApi['inviteStatus']>()
const approve = vi.fn<IClusterApi['respondToJoinRequest']>()
const cancel = vi.fn<IClusterApi['cancelJoinRequest']>()
const getRequests = vi.fn<IClusterApi['getJoinRequests']>()
const respond = vi.fn<IClusterApi['respondToInvite']>()
let container: HTMLDivElement
let root: Root

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        value => value.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing ${label} button`)
    return found
}

async function setAddress(): Promise<void> {
    const input = document.querySelector<HTMLInputElement>('input[placeholder="192.168.1.100"]')
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!input || !setter) throw new Error('Missing address field or setter')
    await act(async () => {
        setter.call(input, '192.0.2.9')
        input.dispatchEvent(new Event('input', { bubbles: true }))
    })
}

beforeEach(() => {
    // Modal blur/focus frames belong to their test, including after unmount.
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] })
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
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
    ask.mockReset().mockResolvedValue(request)
    inviteNode.mockReset().mockResolvedValue(invite)
    status.mockReset().mockResolvedValue(invite)
    approve.mockReset().mockResolvedValue({
        ...request,
        direction: 'incoming',
        state: 'awaiting-requester',
        inviteId: 'child'
    })
    cancel.mockReset().mockResolvedValue({ ...request, state: 'canceled' })
    getRequests.mockReset().mockResolvedValue(baseline)
    respond.mockReset().mockResolvedValue({ ...invite, pin: null, state: 'paired' })
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: {
            cluster: {
                askToJoin: ask,
                inviteNode,
                inviteStatus: status,
                respondToJoinRequest: approve,
                cancelJoinRequest: cancel,
                getJoinRequests: getRequests,
                respondToInvite: respond,
                abandonIfSolo: vi.fn<IClusterApi['abandonIfSolo']>().mockResolvedValue(undefined)
            }
        }
    })
    useConnectionStore.setState({ connected: true, selfId: 'self', clusterId: null })
    useClusterInvitationsStore.setState({
        joinSnapshot: baseline,
        joinError: null,
        joinBusy: {},
        members: [],
        pendingInvites: [],
        activeInviteId: null
    })
    useDiscoveredNodesStore.setState({ nodes: [] })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useClusterInvitationsStore.getState().cleanup()
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

describe('Ask to join entry points', () => {
    it('asks the discovered clustered member from Cluster settings', async () => {
        await act(async () => root.render(<AvailableNodeCard node={peer} />))
        await act(async () => button('Ask to join').click())
        expect(ask).toHaveBeenCalledWith({
            address: peer.ipAddress,
            port: peer.port,
            nodeId: peer.id,
            replaceSoloCluster: false
        })
        expect(inviteNode).not.toHaveBeenCalled()
    })

    it('asks the discovered clustered member from Add node', async () => {
        useDiscoveredNodesStore.setState({ nodes: [peer] })
        await act(async () => root.render(<AddNodeModal open onOpenChange={() => {}} />))
        const choices = Array.from(document.querySelectorAll('button')).filter(
            value => value.textContent?.trim() === 'Ask to join'
        )
        expect(choices).toHaveLength(2)
        const discovered = choices[1]
        if (!discovered) throw new Error('Missing discovered action')
        await act(async () => discovered.click())
        expect(ask).toHaveBeenCalledWith({
            address: peer.ipAddress,
            port: peer.port,
            nodeId: peer.id,
            replaceSoloCluster: false
        })
        expect(inviteNode).not.toHaveBeenCalled()
    })

    it('keeps manual Invite separate from Ask to join', async () => {
        await act(async () => root.render(<AddNodeModal open onOpenChange={() => {}} />))
        await setAddress()
        await act(async () => button('Ask to join').click())
        expect(ask).toHaveBeenCalledWith({
            address: '192.0.2.9',
            port: undefined,
            nodeId: undefined,
            replaceSoloCluster: false
        })
        expect(inviteNode).not.toHaveBeenCalled()
    })

    it('preserves normal manual Invite', async () => {
        await act(async () => root.render(<AddNodeModal open onOpenChange={() => {}} />))
        await setAddress()
        await act(async () => button('Invite').click())
        expect(inviteNode).toHaveBeenCalledWith('192.0.2.9', false)
        expect(ask).not.toHaveBeenCalled()
    })

    it('requires exact solo replacement confirmation and sends nothing on dismissal', async () => {
        useClusterInvitationsStore.setState({
            joinSnapshot: { ...baseline, requiresSoloReplacement: true }
        })
        await act(async () => root.render(<AvailableNodeCard node={peer} />))
        await act(async () => button('Ask to join').click())
        expect(document.body.textContent).toContain(
            "Joining replaces this device's empty cluster. Other devices will not be moved."
        )
        const dialog = document.querySelector('dialog[open]')
        const labelId = dialog?.getAttribute('aria-labelledby')
        if (!labelId) throw new Error('Missing replacement dialog label association')
        expect(document.getElementById(labelId)?.textContent).toBe('Ask to join')
        expect(ask).not.toHaveBeenCalled()
        await act(async () => button('Cancel').click())
        expect(ask).not.toHaveBeenCalled()
        await act(async () => button('Ask to join').click())
        const confirm = Array.from(document.querySelectorAll('dialog button')).find(
            value => value.textContent?.trim() === 'Ask to join'
        )
        if (!(confirm instanceof HTMLButtonElement))
            throw new Error('Missing replacement confirmation')
        await act(async () => confirm.click())
        expect(ask).toHaveBeenCalledWith({
            address: peer.ipAddress,
            port: peer.port,
            nodeId: peer.id,
            replaceSoloCluster: true
        })
    })

    it('explains the authoritative multi-peer refusal', async () => {
        useClusterInvitationsStore.setState({
            joinSnapshot: { ...baseline, canAskToJoin: false, blockedReason: 'already-clustered' }
        })
        await act(async () => root.render(<AvailableNodeCard node={peer} />))
        expect(button('Ask to join').disabled).toBe(true)
        expect(container.textContent).toContain(
            'This device already has cluster peers. Leave its current cluster before asking to join another.'
        )
    })

    it('keeps ordinary Invite available with an old local service', async () => {
        useClusterInvitationsStore.setState({
            joinSnapshot: { ...baseline, supported: false, canAskToJoin: false }
        })
        await act(async () => root.render(<AddNodeModal open onOpenChange={() => {}} />))
        await setAddress()
        expect(button('Ask to join').disabled).toBe(true)
        expect(button('Invite').disabled).toBe(false)
        expect(document.body.textContent).toContain('Update PAIR to ask to join a cluster.')
    })
})

describe('member review and requester progress', () => {
    it('keeps incoming requests passive and remote management blocked by default', async () => {
        const incoming: JoinRequest = { ...request, direction: 'incoming' }
        await act(async () => root.render(<ClusterSettings />))
        const focused = document.createElement('button')
        focused.textContent = 'Existing activity'
        document.body.append(focused)
        focused.focus()
        expect(document.activeElement).toBe(focused)
        await act(async () =>
            useClusterInvitationsStore.setState({
                joinSnapshot: { ...baseline, requests: [incoming] }
            })
        )
        await act(async () => vi.advanceTimersByTimeAsync(32))
        expect(document.querySelector('dialog[open]')).toBeNull()
        expect(document.activeElement).toBe(focused)
        focused.remove()
        expect(container.textContent).toContain('Unverified claimed device')
        await act(async () => button('Approve').click())
        expect(approve).toHaveBeenCalledWith(request.requestId, true, false)
        expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
    })

    it('uses the explicit local grant choice when approving', async () => {
        const incoming: JoinRequest = { ...request, direction: 'incoming' }
        useClusterInvitationsStore.setState({ joinSnapshot: { ...baseline, requests: [incoming] } })
        await act(async () => root.render(<JoinRequestCard request={incoming} />))
        const allow = document.querySelector<HTMLInputElement>('input[type="radio"][value="allow"]')
        if (!allow) throw new Error('Missing remote-management choice')
        await act(async () => allow.click())
        await act(async () => button('Approve').click())
        expect(approve).toHaveBeenCalledWith(request.requestId, true, true)
    })

    it('displays the child PIN only on the approving member', async () => {
        await act(async () =>
            root.render(
                <JoinRequestCard
                    request={{
                        ...request,
                        direction: 'incoming',
                        state: 'pairing',
                        inviteId: 'child'
                    }}
                />
            )
        )
        expect(status).toHaveBeenCalledWith('child')
        expect(container.textContent).toContain('123456')
        expect(document.querySelector('input[aria-label="Pairing PIN"]')).toBeNull()
        await act(async () =>
            root.render(
                <JoinRequestCard request={{ ...request, state: 'pairing', inviteId: 'child' }} />
            )
        )
        expect(container.textContent).not.toContain('123456')
        expect(container.textContent).toContain('Enter the PIN from the selected member')
        expect(container.textContent).not.toContain('Joined the cluster.')
    })

    it('reuses the existing requester PIN-entry prompt for the correlated child', async () => {
        const outgoing: JoinRequest = { ...request, state: 'pairing', inviteId: 'child' }
        status.mockResolvedValue({ ...invite, pin: null })
        getRequests.mockResolvedValue({ ...baseline, requests: [outgoing] })
        await act(async () => useClusterInvitationsStore.getState().refreshJoinRequests())
        expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
        await act(async () =>
            root.render(
                <>
                    <JoinRequestCard request={outgoing} />
                    <InviteApprovalModal />
                </>
            )
        )
        await act(async () => button('Enter PIN').click())
        const input = document.querySelector<HTMLInputElement>('input[aria-label="Pairing PIN"]')
        expect(input?.disabled).toBe(false)
        expect(document.body.textContent).not.toContain('123456')
        expect(document.body.textContent).not.toContain('Joined the cluster.')
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        if (!input || !setter) throw new Error('Missing recovered PIN field or setter')
        await act(async () => {
            setter.call(input, '123456')
            input.dispatchEvent(new Event('input', { bubbles: true }))
        })
        await act(async () => button('Join cluster').click())
        expect(respond).toHaveBeenCalledWith('child', true, '123456', false)
    })

    it('does not display a PIN or success when exact child status is lost', async () => {
        status.mockRejectedValue(new Error('unknown invite'))
        await act(async () =>
            root.render(
                <JoinRequestCard
                    request={{
                        ...request,
                        direction: 'incoming',
                        state: 'pairing',
                        inviteId: 'child'
                    }}
                />
            )
        )
        expect(status).toHaveBeenCalledWith('child')
        expect(container.textContent).toContain('Joining is not confirmed')
        expect(container.textContent).not.toContain('123456')
        expect(container.textContent).not.toContain('Joined the cluster.')
    })

    it('coalesces approvals from two attached views of the same request', async () => {
        const incoming: JoinRequest = { ...request, direction: 'incoming' }
        useClusterInvitationsStore.setState({ joinSnapshot: { ...baseline, requests: [incoming] } })
        await act(async () =>
            root.render(
                <>
                    <JoinRequestCard request={incoming} />
                    <JoinRequestCard request={incoming} />
                </>
            )
        )
        const approvals = Array.from(container.querySelectorAll('button')).filter(
            value => value.textContent === 'Approve'
        )
        await act(async () => {
            for (const approval of approvals) approval.click()
        })
        expect(approve).toHaveBeenCalledTimes(1)
    })

    it('declines the addressed request without issuing an invitation', async () => {
        const incoming: JoinRequest = { ...request, direction: 'incoming' }
        useClusterInvitationsStore.setState({ joinSnapshot: { ...baseline, requests: [incoming] } })
        approve.mockResolvedValue({ ...incoming, state: 'declined' })
        await act(async () => root.render(<JoinRequestCard request={incoming} />))
        await act(async () => button('Decline').click())
        expect(approve).toHaveBeenCalledWith(request.requestId, false)
        expect(inviteNode).not.toHaveBeenCalled()
    })

    it('cancels explicitly while waiting and renders the terminal outcome', async () => {
        await act(async () => root.render(<JoinRequestCard request={request} />))
        expect(container.textContent).toContain('Waiting for approval on Member')
        expect(container.textContent).toContain('Open PAIR on that device, then Settings > Cluster')
        await act(async () => button('Cancel request').click())
        expect(cancel).toHaveBeenCalledWith(request.requestId)
        await act(async () =>
            root.render(<JoinRequestCard request={{ ...request, state: 'canceled' }} />)
        )
        expect(container.textContent).toContain('Join request canceled.')
        expect(
            Array.from(container.querySelectorAll('button')).some(
                value => value.textContent === 'Cancel request'
            )
        ).toBe(false)
    })

    it.each([
        'declined',
        'expired',
        'unsupported-peer',
        'approver-changed',
        'request-status-lost',
        'connection-refused'
    ])('explains %s without claiming joining succeeded', reason => {
        const message = joinRequestMessage({
            ...request,
            state: reason === 'declined' ? 'declined' : reason === 'expired' ? 'expired' : 'failed',
            reason,
            inviteId: 'child'
        })
        expect(message).not.toContain('Joined the cluster.')
        expect(message.length).toBeGreaterThan(20)
        if (reason === 'connection-refused') expect(message).toContain('firewall')
    })

    it('keeps pairing and paired copy on the correct side of the ceremony', () => {
        expect(joinRequestMessage({ ...request, direction: 'incoming', state: 'pairing' })).toBe(
            'Preparing to pair. Read the PIN displayed on this member and enter it on the requesting device.'
        )
        expect(joinRequestMessage({ ...request, direction: 'outgoing', state: 'pairing' })).toBe(
            'Preparing to pair. Enter the PIN from the selected member in the pairing prompt.'
        )
        expect(joinRequestMessage({ ...request, direction: 'incoming', state: 'paired' })).toBe(
            'The requesting device was admitted to this cluster.'
        )
        expect(joinRequestMessage({ ...request, direction: 'outgoing', state: 'paired' })).toBe(
            'Joined the cluster.'
        )
    })

    it('does not infer connection-failure direction from the presence of a child ID', () => {
        const expected =
            "The join connection failed. Check the address, pairing port, and that PAIR is running on both devices. The member must also reach the requesting device's pairing port. Check firewalls on both devices and local-network reachability."
        expect(
            joinRequestMessage({
                ...request,
                state: 'failed',
                reason: 'connection-timeout',
                inviteId: null
            })
        ).toBe(expected)
        expect(
            joinRequestMessage({
                ...request,
                state: 'failed',
                reason: 'connection-timeout',
                inviteId: 'allocated-before-polling'
            })
        ).toBe(expected)
        expect(
            joinRequestMessage({
                ...request,
                direction: 'incoming',
                state: 'failed',
                reason: 'connection-refused',
                inviteId: 'child'
            })
        ).toBe(expected)
    })
})
