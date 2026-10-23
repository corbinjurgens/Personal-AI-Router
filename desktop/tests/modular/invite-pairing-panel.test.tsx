// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Invite, RemoteManagementSnapshot } from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { InvitePairingPanel, RemoteManagementChoice } from '@/ui/components/InvitePairingPanel'
import { useInvitePairing } from '@/ui/hooks/useInvitePairing'
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
    pin: '123456',
    state: 'failed',
    reason: '',
    createdAt: 1,
    respondedAt: null
}
let container: HTMLDivElement
let root: Root
const reset = vi.fn<() => void>()
beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useClusterInvitationsStore.getState().cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

describe('failed invitation recovery', () => {
    it('shows the later authoritative denial instead of the submitted allowance', async () => {
        let push: ((snapshot: RemoteManagementSnapshot) => void) | undefined
        const inviteNode = vi.fn<IClusterApi['inviteNode']>().mockResolvedValue({
            ...invite,
            state: 'paired'
        })
        const getRemoteManagement = vi.fn<IClusterApi['getRemoteManagement']>().mockResolvedValue({
            version: 1,
            revision: '1',
            grants: []
        })
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                cluster: {
                    getJoinRequests: vi.fn<IClusterApi['getJoinRequests']>().mockResolvedValue({
                        supported: false,
                        canAskToJoin: false,
                        requiresSoloReplacement: false,
                        blockedReason: 'unsupported',
                        requests: []
                    }),
                    onJoinRequestsChanged: vi
                        .fn<IClusterApi['onJoinRequestsChanged']>()
                        .mockReturnValue(() => {}),
                    inviteNode,
                    getRemoteManagement,
                    onRemoteManagementChanged: (
                        callback: (snapshot: RemoteManagementSnapshot) => void
                    ) => {
                        push = callback
                        return () => {}
                    },
                    onInviteReceived: () => () => {},
                    onPendingInvitesChanged: () => () => {}
                },
                nodes: { onMembersChanged: () => () => {} }
            }
        })
        await useClusterInvitationsStore.getState().initialize({
            info: { clusterId: 'cluster', isClustered: true, clusterFriendlyName: 'Home' },
            identity: {
                nodeUuid: 'self',
                nodeId: 'self',
                name: 'Self',
                certFingerprint: 'cert',
                clusterId: 'cluster'
            },
            pendingInvites: [],
            members: [
                {
                    id: 'peer',
                    nodeUuid: 'paired-peer',
                    name: 'Peer',
                    ipAddress: '127.0.0.1',
                    port: 1,
                    clusterId: 'cluster',
                    state: 'member',
                    joinedAt: 1,
                    lastSeen: 1
                }
            ]
        })
        function Harness() {
            const pairing = useInvitePairing()
            return (
                <>
                    <RemoteManagementChoice
                        allowed={pairing.allowRemoteManagement}
                        onChange={pairing.setAllowRemoteManagement}
                    />
                    <button id="start" onClick={() => void pairing.start('127.0.0.1')}>
                        Start
                    </button>
                    <InvitePairingPanel
                        invite={pairing.invite}
                        error={pairing.error}
                        onReset={pairing.reset}
                        savedPermission={pairing.savedPermission}
                    />
                </>
            )
        }
        await act(async () => root.render(<Harness />))
        await act(async () =>
            container.querySelector<HTMLInputElement>('input[value="allow"]')?.click()
        )
        await act(async () => container.querySelector<HTMLButtonElement>('#start')?.click())
        expect(inviteNode).toHaveBeenCalledWith('127.0.0.1', true)
        expect(container.textContent).toContain('Reading the saved remote-management permission')
        expect(container.textContent).not.toContain('Pairing saved:')
        if (!push) throw new Error('Missing permission push subscription')
        await act(async () =>
            push?.({
                version: 1,
                revision: '2',
                grants: [
                    { peerUuid: 'unrelated-peer', allowed: true, state: 'allowed' },
                    { peerUuid: 'paired-peer', allowed: false, state: 'denied' }
                ]
            })
        )
        expect(container.textContent).toContain('is blocked from changing this device')
        expect(container.textContent).not.toContain('may manage this device')
        expect(container.textContent).toContain(
            'The other device chooses its own permission independently'
        )
    })
    it.each(['rollback-pending', 'awaiting-confirmation'])(
        'shows %s without requesting another PIN entry',
        async reason => {
            await act(async () =>
                root.render(
                    <InvitePairingPanel
                        invite={{ ...invite, state: 'pending', reason }}
                        error={null}
                        onReset={reset}
                    />
                )
            )
            expect(container.textContent).toContain(
                reason === 'rollback-pending'
                    ? 'Waiting for cleanup'
                    : 'Waiting for pairing confirmation'
            )
            expect(container.textContent).not.toContain('123456')
            expect(container.textContent).not.toContain('Enter this PIN')
        }
    )
    it.each([
        'connection-reset',
        'connection-refused',
        'connection-timeout',
        'connection-unreachable'
    ])('retains the diagnostic and gives reverse invitation guidance for %s', async reason => {
        await act(async () =>
            root.render(
                <InvitePairingPanel
                    invite={{ ...invite, reason, error: 'actual transport diagnostic' }}
                    error={null}
                    onReset={reset}
                />
            )
        )
        expect(container.textContent).toContain('actual transport diagnostic')
        expect(container.textContent).toContain(
            'If the other device blocks incoming connections, open PAIR on that device and invite this computer instead.'
        )
        const button = Array.from(container.querySelectorAll('button')).find(
            value => value.textContent === 'Try again'
        )
        expect(button?.textContent).toBe('Try again')
        await act(async () => button?.click())
        expect(reset).toHaveBeenCalledTimes(1)
    })
    it.each(['incorrect-pin', '', 'protocol-failure', 'already-clustered'])(
        'does not suggest reverse invitation for %s',
        async reason => {
            await act(async () =>
                root.render(
                    <InvitePairingPanel
                        invite={{
                            ...invite,
                            reason,
                            state: reason === 'already-clustered' ? 'rejected' : 'failed'
                        }}
                        error={null}
                        onReset={reset}
                    />
                )
            )
            expect(container.textContent).not.toContain('blocks incoming connections')
            expect(container.textContent).toContain(
                reason === 'incorrect-pin'
                    ? 'Incorrect PIN'
                    : reason === 'already-clustered'
                      ? 'already in a cluster'
                      : 'Pairing failed'
            )
        }
    )
    it('keeps explicit refusal separate from connection refusal', async () => {
        await act(async () =>
            root.render(
                <InvitePairingPanel
                    invite={{ ...invite, state: 'rejected', reason: 'user-refused' }}
                    error={null}
                    onReset={reset}
                />
            )
        )
        expect(container.textContent).toContain('The other node refused the invite.')
        expect(container.textContent).not.toContain('blocks incoming connections')
    })
    it('retains cancel while a PIN is pending', async () => {
        const cancel = vi.fn<() => void>()
        await act(async () =>
            root.render(
                <InvitePairingPanel
                    invite={{ ...invite, state: 'pending' }}
                    error={null}
                    onReset={reset}
                    onCancel={cancel}
                />
            )
        )
        expect(container.textContent).toContain('123456')
        await act(async () => container.querySelector('button')?.click())
        expect(cancel).toHaveBeenCalledTimes(1)
    })
})

describe('outbound pairing hook lifecycle', () => {
    it.each([false, true])(
        'sends the selected local permission %s and resets to Block',
        async allowed => {
            const inviteNode = vi
                .fn<IClusterApi['inviteNode']>()
                .mockResolvedValue({ ...invite, state: 'paired' })
            vi.stubGlobal('window', { pairApi: { cluster: { inviteNode } } })
            function Harness() {
                const pairing = useInvitePairing()
                return (
                    <>
                        <RemoteManagementChoice
                            allowed={pairing.allowRemoteManagement}
                            onChange={pairing.setAllowRemoteManagement}
                            disabled={pairing.submitting}
                        />
                        <button id="start" onClick={() => void pairing.start('127.0.0.1')}>
                            Start
                        </button>
                        <button id="reset" onClick={pairing.reset}>
                            Reset
                        </button>
                    </>
                )
            }
            await act(async () => root.render(<Harness />))
            expect(container.querySelector<HTMLInputElement>('input[value="block"]')?.checked).toBe(
                true
            )
            if (allowed)
                await act(async () =>
                    container.querySelector<HTMLInputElement>('input[value="allow"]')?.click()
                )
            await act(async () => container.querySelector<HTMLButtonElement>('#start')?.click())
            expect(inviteNode).toHaveBeenCalledWith('127.0.0.1', allowed)
            await act(async () => container.querySelector<HTMLButtonElement>('#reset')?.click())
            expect(container.querySelector<HTMLInputElement>('input[value="block"]')?.checked).toBe(
                true
            )
        }
    )

    it.each(['resolve', 'reject'])(
        'preserves replacement B and its polling after late cancellation A %s',
        async outcome => {
            vi.useFakeTimers()
            const first = { ...invite, inviteId: 'A', state: 'pending' } satisfies Invite
            const second = { ...first, inviteId: 'B' }
            let finishCancel: ((value: Invite) => void) | undefined
            let failCancel: ((error: Error) => void) | undefined
            const cancelInvite = vi.fn<IClusterApi['cancelInvite']>().mockImplementation(
                () =>
                    new Promise((resolve, reject) => {
                        finishCancel = resolve
                        failCancel = reject
                    })
            )
            const inviteNode = vi
                .fn<IClusterApi['inviteNode']>()
                .mockResolvedValueOnce(first)
                .mockResolvedValueOnce(second)
            const inviteStatus = vi.fn<IClusterApi['inviteStatus']>().mockResolvedValue(second)
            const abandonIfSolo = vi.fn<IClusterApi['abandonIfSolo']>().mockResolvedValue()
            vi.stubGlobal('window', {
                pairApi: { cluster: { inviteNode, cancelInvite, inviteStatus, abandonIfSolo } }
            })
            function Harness() {
                const pairing = useInvitePairing()
                return (
                    <>
                        <button id="start" onClick={() => void pairing.start('127.0.0.1')}>
                            Start
                        </button>
                        <button id="cancel" onClick={() => void pairing.cancel()}>
                            Cancel
                        </button>
                        <button id="reset" onClick={pairing.reset}>
                            Reset
                        </button>
                        <output>{pairing.invite?.inviteId}</output>
                    </>
                )
            }
            await act(async () => root.render(<Harness />))
            await act(async () => container.querySelector<HTMLButtonElement>('#start')?.click())
            expect(container.querySelector('output')?.textContent).toBe('A')
            await act(async () => container.querySelector<HTMLButtonElement>('#cancel')?.click())
            expect(cancelInvite).toHaveBeenCalledWith('A')
            await act(async () => container.querySelector<HTMLButtonElement>('#reset')?.click())
            await act(async () => container.querySelector<HTMLButtonElement>('#start')?.click())
            expect(container.querySelector('output')?.textContent).toBe('B')
            expect(finishCancel).toBeTypeOf('function')
            expect(failCancel).toBeTypeOf('function')
            await act(async () => {
                if (outcome === 'resolve') finishCancel?.({ ...first, state: 'canceled' })
                else failCancel?.(new Error('connection reset'))
            })
            expect(container.querySelector('output')?.textContent).toBe('B')
            expect(abandonIfSolo).not.toHaveBeenCalled()
            await act(async () =>
                vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS)
            )
            expect(inviteStatus).toHaveBeenCalledWith('B')
            await act(async () => root.render(null))
            await act(async () =>
                vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
            )
            expect(inviteStatus).toHaveBeenCalledTimes(1)
        }
    )
    it('renders the initial classified failure from the actual hook', async () => {
        const inviteNode = vi
            .fn<IClusterApi['inviteNode']>()
            .mockResolvedValue({ ...invite, reason: 'connection-reset', error: 'socket closed' })
        const abandonIfSolo = vi.fn<IClusterApi['abandonIfSolo']>().mockResolvedValue()
        vi.stubGlobal('window', { pairApi: { cluster: { inviteNode, abandonIfSolo } } })
        function Harness() {
            const pairing = useInvitePairing()
            return (
                <>
                    <button onClick={() => void pairing.start('127.0.0.1')}>Start</button>
                    <InvitePairingPanel
                        invite={pairing.invite}
                        error={pairing.error}
                        onReset={pairing.reset}
                    />
                </>
            )
        }
        await act(async () => root.render(<Harness />))
        await act(async () => container.querySelector('button')?.click())
        expect(container.textContent).toContain('socket closed')
        expect(container.textContent).toContain('blocks incoming connections')
        expect(abandonIfSolo).toHaveBeenCalledTimes(1)
    })
    it('ignores stale polls after reset and stops polling on unmount', async () => {
        vi.useFakeTimers()
        let finish: ((value: Invite) => void) | undefined
        const inviteNode = vi
            .fn<IClusterApi['inviteNode']>()
            .mockResolvedValue({ ...invite, state: 'pending' })
        const inviteStatus = vi.fn<IClusterApi['inviteStatus']>().mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        vi.stubGlobal('window', { pairApi: { cluster: { inviteNode, inviteStatus } } })
        function Harness() {
            const pairing = useInvitePairing()
            return (
                <>
                    <button onClick={() => void pairing.start('127.0.0.1')}>Start</button>
                    <button onClick={pairing.reset}>Reset</button>
                    <span>{pairing.invite?.state}</span>
                </>
            )
        }
        await act(async () => root.render(<Harness />))
        await act(async () => container.querySelector('button')?.click())
        await act(async () => vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS))
        expect(inviteStatus).toHaveBeenCalledWith('one')
        const resetButton = Array.from(container.querySelectorAll('button')).find(
            button => button.textContent === 'Reset'
        )
        await act(async () => resetButton?.click())
        expect(finish).toBeTypeOf('function')
        await act(async () => finish?.({ ...invite, state: 'paired' }))
        expect(container.textContent).not.toContain('paired')
        await act(async () => root.render(null))
        await act(async () =>
            vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 3)
        )
        expect(inviteStatus).toHaveBeenCalledTimes(1)
    })
})
