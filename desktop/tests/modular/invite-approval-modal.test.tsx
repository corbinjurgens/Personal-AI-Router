// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Invite } from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { InviteApprovalModal } from '@/ui/components/InviteApprovalModal'
import PendingInviteCard from '@/ui/components/ClusterSettings/PendingInviteCard'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'

const invite: Invite = {
    inviteId: 'one',
    fromNodeId: 'sender',
    fromNodeUuid: 'uuid',
    fromNodeName: 'Sender',
    toNodeId: null,
    clusterId: 'cluster',
    clusterFriendlyName: 'Home',
    pin: null,
    state: 'pending',
    reason: '',
    createdAt: 1,
    respondedAt: null
}
const respond = vi.fn<IClusterApi['respondToInvite']>()
const status = vi.fn<IClusterApi['inviteStatus']>()
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

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        value => value.textContent === label
    )
    if (!found) throw new Error(`Missing ${label} button`)
    return found
}

async function submitPIN(): Promise<void> {
    const input = document.querySelector('input[aria-label="Pairing PIN"]')
    if (!input) throw new Error('Missing PIN input')
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing input value setter')
    await act(async () => {
        setter.call(input, '123456')
        input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => button('Join cluster').click())
}

beforeEach(async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.useFakeTimers()
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { cluster: { respondToInvite: respond, inviteStatus: status } }
    })
    status.mockResolvedValue(invite)
    useClusterInvitationsStore.setState({
        pendingInvites: [invite],
        activeInviteId: 'one',
        remoteManagement: {
            version: 1,
            revision: '1',
            grants: [{ peerUuid: 'uuid', allowed: false, state: 'denied' }]
        }
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
    await act(async () => root.render(<InviteApprovalModal />))
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useClusterInvitationsStore.getState().cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

describe('receiver pending pairing', () => {
    it('focuses the PIN while the management choices remain keyboard inputs', async () => {
        await act(async () => vi.advanceTimersByTimeAsync(20))
        expect(document.activeElement?.getAttribute('aria-label')).toBe('Pairing PIN')
        expect(document.querySelectorAll('input[type="radio"]')).toHaveLength(2)
    })
    it.each([false, true])('submits the local management choice %s with the PIN', async allowed => {
        respond.mockResolvedValue({ ...invite, reason: 'awaiting-confirmation' })
        const blocked = document.querySelector<HTMLInputElement>(
            'input[type="radio"][value="block"]'
        )
        expect(blocked?.checked).toBe(true)
        if (allowed) {
            const allow = document.querySelector<HTMLInputElement>(
                'input[type="radio"][value="allow"]'
            )
            if (!allow) throw new Error('Missing Allow choice')
            await act(async () => allow.click())
        }
        await submitPIN()
        expect(respond).toHaveBeenCalledWith('one', true, '123456', allowed)
        expect(document.body.textContent).toContain('Remote management of this device')
        expect(document.body.textContent).toContain('Local pause and engine Off always win')
    })

    it('resets the permission and PIN for a different invitation', async () => {
        const allow = document.querySelector<HTMLInputElement>('input[type="radio"][value="allow"]')
        if (!allow) throw new Error('Missing Allow choice')
        await act(async () => allow.click())
        await act(async () =>
            useClusterInvitationsStore.setState({
                pendingInvites: [{ ...invite, inviteId: 'two' }],
                activeInviteId: 'two'
            })
        )
        expect(
            document.querySelector<HTMLInputElement>('input[type="radio"][value="block"]')?.checked
        ).toBe(true)
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Pairing PIN"]')?.value
        ).toBe('')
    })

    it('declines without sending a management choice', async () => {
        respond.mockResolvedValue({ ...invite, state: 'declined' })
        await act(async () => button('Decline').click())
        expect(respond).toHaveBeenCalledWith('one', false)
    })

    it('keeps a confirmed pairing visible without inventing an unavailable saved permission', async () => {
        await act(async () =>
            useClusterInvitationsStore.setState({
                inviteResults: { one: { ...invite, state: 'paired' } },
                remoteManagement: { version: 1, revision: '', grants: [] },
                remoteManagementLoading: false
            })
        )
        expect(document.body.textContent).toContain('Node paired successfully')
        expect(document.body.textContent).toContain('saved permission is unavailable')
        expect(document.body.textContent).not.toContain('may manage this device')
        expect(document.body.textContent).not.toContain('is blocked from changing this device')
    })

    it('keeps saved confirmation when matching terminal status arrives before the pending command response', async () => {
        let finish: ((value: Invite) => void) | undefined
        respond.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        status.mockResolvedValue({ ...invite, state: 'paired' })
        await submitPIN()
        expect(useClusterInvitationsStore.getState().activeInviteId).toBe('one')
        expect(document.body.textContent).toContain('is blocked from changing this device')
        expect(finish).toBeTypeOf('function')
        await act(async () => finish?.(invite))
        expect(document.body.textContent).toContain('Node paired successfully')
        expect(document.body.textContent).not.toContain('Waiting for pairing confirmation')
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('paired')
        await act(async () => button('Close').click())
        expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
    })

    it('preserves the replacement modal when the previous invitation reaches terminal status', async () => {
        respond.mockResolvedValue(invite)
        await submitPIN()
        const second = { ...invite, inviteId: 'two', fromNodeName: 'Second' }
        await act(async () =>
            useClusterInvitationsStore.setState({
                pendingInvites: [invite, second],
                activeInviteId: 'two'
            })
        )
        const calls = status.mock.calls.length
        status.mockResolvedValue({ ...invite, state: 'failed', reason: 'storage-write-failed' })
        await act(async () =>
            vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
        )
        expect(status).toHaveBeenCalledTimes(calls + 1)
        expect(document.body.textContent).toContain('Second')
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Pairing PIN"]')?.disabled
        ).toBe(false)
        expect(document.body.textContent).not.toContain('storage-write-failed')
        expect(useClusterInvitationsStore.getState().activeInviteId).toBe('two')
    })
    it('keeps pending visible, disables another PIN, and confirms only matching paired status', async () => {
        respond.mockResolvedValue({ ...invite, reason: 'awaiting-confirmation' })
        await submitPIN()
        expect(document.body.textContent).toContain('Waiting for pairing confirmation')
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Pairing PIN"]')?.disabled
        ).toBe(true)
        expect(document.querySelectorAll('input[type="radio"]')).toHaveLength(2)
        expect(
            Array.from(document.querySelectorAll<HTMLInputElement>('input[type="radio"]')).every(
                input => input.disabled
            )
        ).toBe(true)
        await act(async () =>
            document
                .querySelector('input[aria-label="Pairing PIN"]')
                ?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
        )
        expect(respond).toHaveBeenCalledTimes(1)
        status.mockResolvedValueOnce({ ...invite, inviteId: 'other', state: 'paired' })
        await act(async () => vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS))
        expect(document.body.textContent).toContain('Waiting for pairing confirmation')
        status.mockResolvedValueOnce({ ...invite, state: 'paired' })
        await act(async () => vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS))
        expect(useClusterInvitationsStore.getState().activeInviteId).toBe('one')
        expect(document.body.textContent).toContain('Node paired successfully')
        expect(document.body.textContent).not.toContain('Waiting for pairing confirmation')
        expect(status).toHaveBeenCalledWith('one')
    })

    it.each(['disk-write-failed', 'incorrect-pin'])(
        'shows terminal %s accurately',
        async reason => {
            respond.mockResolvedValue({ ...invite, state: 'failed', reason })
            await submitPIN()
            expect(document.body.textContent).toContain(
                reason === 'incorrect-pin' ? 'Incorrect PIN.' : 'Pairing failed: disk-write-failed'
            )
            if (reason !== 'incorrect-pin')
                expect(document.body.textContent).not.toContain('Incorrect PIN')
            expect(document.querySelector('input')).toBeNull()
            await act(async () => button('Close').click())
            expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
        }
    )

    it('keeps one store-owned cadence across close, reopen and unmount until store cleanup', async () => {
        status.mockResolvedValue({ ...invite, reason: 'rollback-pending' })
        respond.mockResolvedValue({ ...invite, reason: 'rollback-pending' })
        await submitPIN()
        expect(document.body.textContent).toContain('Waiting for cleanup')
        await act(async () => button('Close').click())
        const calls = status.mock.calls.length
        await act(async () =>
            vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
        )
        expect(status).toHaveBeenCalledTimes(calls + 3)
        await act(async () =>
            root.render(
                <>
                    <InviteApprovalModal />
                    <PendingInviteCard invite={invite} />
                </>
            )
        )
        expect(button('Decline').disabled).toBe(true)
        await act(async () => button('View pairing status').click())
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Pairing PIN"]')?.disabled
        ).toBe(true)
        expect(respond).toHaveBeenCalledTimes(1)
        await act(async () => root.render(null))
        const unmountedCalls = status.mock.calls.length
        await act(async () =>
            vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
        )
        expect(status).toHaveBeenCalledTimes(unmountedCalls + 3)
        useClusterInvitationsStore.getState().cleanup()
        const cleanedCalls = status.mock.calls.length
        await act(async () =>
            vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
        )
        expect(status).toHaveBeenCalledTimes(cleanedCalls)
        expect(vi.getTimerCount()).toBe(0)
    })

    it('reports unknown status honestly and stops polling without wrong-PIN copy', async () => {
        respond.mockResolvedValue({ ...invite, reason: 'awaiting-confirmation' })
        await submitPIN()
        status.mockRejectedValueOnce(new Error('-32001: unknown inviteId'))
        await act(async () => vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS))
        expect(document.body.textContent).toContain('Pairing could not be confirmed')
        expect(document.body.textContent).not.toContain('Incorrect PIN')
        expect(document.querySelector('input')).toBeNull()
    })
})
