// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ClusterNode, Invite, RemoteManagementSnapshot } from '@/shared/types/cluster'
import { INVALID_PARAMS_CODE, POLICY_CONFLICT_CODE } from '@/shared/constants/service-errors'
import { getPairTransport } from '@/ui/api/bootstrap'
import { serviceErrorCode } from '@/ui/api/service-transport'
import getErrorString from '@/shared/utils/get-error-string'
import { isInviteStatusUnknown } from '@/ui/utils/cluster-invite-error'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'

interface RemoteManagementChange {
    allowed: boolean
    phase: 'saving' | 'saved' | 'conflict' | 'failed' | 'unknown'
    message?: string
}

interface ClusterInvitationsStore {
    /**
     * Every live inbound invite awaiting the local user's PIN entry. Electron
     * main is authoritative: this list is hydrated from `cluster:get-initial` and
     * updated on every `cluster:pending-invites-changed` snapshot. Submitted
     * invites remain visible until an exact-invite status establishes their outcome.
     */
    pendingInvites: Invite[]
    /** The invite currently surfaced in the approval modal, if any. */
    activeInviteId: string | null
    /** Full cluster membership (members + pending), driven by `nodes:changed`. */
    members: ClusterNode[]
    /** Submitted PINs and their exact-invite outcomes, retained across modal dismissal. */
    inviteResults: Record<string, Invite>
    remoteManagement: RemoteManagementSnapshot | null
    remoteManagementLoading: boolean
    remoteManagementError: string | null
    remoteManagementChanges: Record<string, RemoteManagementChange>
    /** Current identity/membership generation; null until a cluster baseline exists. */
    remoteManagementContext: number | null
    refreshRemoteManagement: () => Promise<void>
    setRemoteManagement: (peerUuid: string, allowed: boolean) => Promise<void>
    initialize: (initial?: ClusterInitialSnapshot) => Promise<void>
    hydrate: (snapshot: ClusterInitialSnapshot) => void
    refresh: () => Promise<void>
    cleanup: () => void
    /** Surface a specific pending invite in the approval modal (e.g. from the card). */
    setActiveInvite: (inviteId: string) => void
    /** Close the approval modal without responding. */
    clearActiveInvite: () => void
    /** Accept (with PIN) or decline a specific inbound invite. */
    respondToInvite: (
        inviteId: string,
        ...response: [accept: true, pin: string, allowRemoteManagement: boolean] | [accept: false]
    ) => Promise<Invite>
    pollInvite: (inviteId: string) => Promise<void>
}

let unsubs: Array<() => void> = []

/** Drop `activeInviteId` if the invite it points at is no longer pending. */
function stillActive(activeInviteId: string | null, invites: Invite[]): string | null {
    if (!activeInviteId) return null
    return invites.some(invite => invite.inviteId === activeInviteId) ? activeInviteId : null
}

function pendingWithResults(invites: Invite[], results: Record<string, Invite>): Invite[] {
    const pending = invites.filter(
        invite => !results[invite.inviteId] || results[invite.inviteId].state === 'pending'
    )
    for (const result of Object.values(results)) {
        if (
            result.state === 'pending' &&
            !pending.some(invite => invite.inviteId === result.inviteId)
        )
            pending.push(result)
    }
    return pending.map(invite => results[invite.inviteId] ?? invite)
}

let generation = 0
let permissionGeneration = 0
let permissionRevision = ''
let permissionNeedsBaseline = false
let localIdentity = ''
let connected = true
let permissionRead: { generation: number; request: Promise<void> } | null = null
// Permission pushes carry no identity. Once this renderer has crossed an
// identity boundary, only a current read can establish authority again.
let permissionPushRequiresReadback = false
let permissionPushReadback: object | null = null

function invalidatePermissions(resetRevision = false): void {
    permissionGeneration++
    if (resetRevision) {
        permissionRevision = ''
        permissionPushReadback = null
    }
    useClusterInvitationsStore.setState(state => {
        const remoteManagementChanges = { ...state.remoteManagementChanges }
        for (const [peerUuid, change] of Object.entries(remoteManagementChanges))
            if (change.phase === 'saving')
                remoteManagementChanges[peerUuid] = {
                    ...change,
                    phase: 'unknown',
                    message:
                        'The device changed before the save was confirmed. Read back before trying again.'
                }
        return {
            remoteManagement: null,
            remoteManagementLoading: false,
            remoteManagementError: 'Remote management permissions are unavailable.',
            remoteManagementChanges
        }
    })
}

function queuePermissionReadback(): void {
    if (permissionPushReadback) return
    const queued = {}
    permissionPushReadback = queued
    // Coalesce notifications behind one current read and one follow-up.
    // Another follow-up requires a new notification, never a failed read.
    void (permissionRead?.request ?? Promise.resolve()).then(() => {
        if (permissionPushReadback !== queued || !connected) return
        permissionPushReadback = null
        void useClusterInvitationsStore.getState().refreshRemoteManagement()
    })
}

function receivePermissions(snapshot: RemoteManagementSnapshot, baseline = false): boolean {
    if (snapshot.revision === '') {
        invalidatePermissions()
        return false
    }
    if (permissionNeedsBaseline && !baseline) return false
    if (permissionRevision && BigInt(snapshot.revision) < BigInt(permissionRevision)) return false
    permissionNeedsBaseline = false
    permissionRevision = snapshot.revision
    useClusterInvitationsStore.setState({
        remoteManagement: snapshot,
        remoteManagementError: null
    })
    return true
}

function membershipKey(members: ClusterNode[]): string {
    return JSON.stringify(
        members.map(member => [member.nodeUuid, member.state, member.joinedAt]).sort()
    )
}

function receiveMembers(members: ClusterNode[], firstBaseline = false): void {
    const store = useClusterInvitationsStore.getState()
    const changed = membershipKey(members) !== membershipKey(store.members)
    useClusterInvitationsStore.setState({ members })
    if (changed) {
        invalidatePermissions()
        permissionNeedsBaseline = !firstBaseline
        useClusterInvitationsStore.setState({
            remoteManagementContext: localIdentity ? permissionGeneration : null
        })
        if (connected) void store.refreshRemoteManagement()
    }
}

interface InvitePoll {
    interval: ReturnType<typeof setInterval>
    request: Promise<void> | null
}

const invitePolls = new Map<string, InvitePoll>()

function ensurePolling(inviteId: string): InvitePoll {
    const existing = invitePolls.get(inviteId)
    if (existing) return existing
    const poll: InvitePoll = {
        interval: setInterval(
            () => void useClusterInvitationsStore.getState().pollInvite(inviteId),
            MODULAR_INVITE_STATUS_POLL_INTERVAL_MS
        ),
        request: null
    }
    invitePolls.set(inviteId, poll)
    return poll
}

function stopPolling(inviteId: string): void {
    const poll = invitePolls.get(inviteId)
    if (poll) clearInterval(poll.interval)
    invitePolls.delete(inviteId)
}

function applyResult(result: Invite): void {
    useClusterInvitationsStore.setState(state => {
        // A delayed pending command or poll cannot overwrite a terminal outcome.
        if (state.inviteResults[result.inviteId]?.state !== 'pending') return state
        const inviteResults = { ...state.inviteResults, [result.inviteId]: result }
        return {
            inviteResults,
            pendingInvites: pendingWithResults(state.pendingInvites, inviteResults)
        }
    })
    if (result.state !== 'pending') stopPolling(result.inviteId)
}

export const useClusterInvitationsStore = create<ClusterInvitationsStore>((set, get) => ({
    pendingInvites: [],
    activeInviteId: null,
    members: [],
    inviteResults: {},
    remoteManagement: null,
    remoteManagementLoading: false,
    remoteManagementError: null,
    remoteManagementChanges: {},
    remoteManagementContext: null,

    hydrate: snapshot => {
        const firstBaseline = localIdentity === ''
        const identity = JSON.stringify([
            snapshot.identity.nodeUuid,
            snapshot.identity.certFingerprint,
            snapshot.identity.clusterId
        ])
        const identityChanged = identity !== localIdentity
        const membersChanged = membershipKey(snapshot.members) !== membershipKey(get().members)
        if (identityChanged) {
            if (!firstBaseline) permissionPushRequiresReadback = true
            localIdentity = identity
            invalidatePermissions(true)
            permissionNeedsBaseline = !firstBaseline
        }
        receiveMembers(snapshot.members, firstBaseline)
        set(state => ({
            pendingInvites: pendingWithResults(snapshot.pendingInvites, state.inviteResults),
            members: snapshot.members,
            remoteManagementContext: permissionGeneration,
            activeInviteId:
                state.activeInviteId && state.inviteResults[state.activeInviteId]
                    ? state.activeInviteId
                    : stillActive(state.activeInviteId, snapshot.pendingInvites)
        }))
        if (identityChanged && !membersChanged && connected) void get().refreshRemoteManagement()
    },

    initialize: async initial => {
        generation++
        invalidatePermissions(true)
        permissionNeedsBaseline = false
        localIdentity = ''
        set({ remoteManagementContext: null })
        unsubs.forEach(stop => stop())
        unsubs = []
        for (const inviteId of invitePolls.keys()) stopPolling(inviteId)
        for (const result of Object.values(get().inviteResults))
            if (result.state === 'pending') ensurePolling(result.inviteId)
        if (!window.pairApi) return
        const transport = getPairTransport()
        const subscriptionGeneration = generation
        connected = transport?.connected ?? true
        unsubs.push(
            window.pairApi.cluster.onRemoteManagementChanged(snapshot => {
                if (!connected || subscriptionGeneration !== generation) return
                if (permissionPushRequiresReadback) {
                    invalidatePermissions()
                    queuePermissionReadback()
                } else receivePermissions(snapshot)
            }),
            // A fresh arrival: surface it in the modal. Main also emits the full
            // list via `cluster:pending-invites-changed`, so this only picks which
            // invite the modal shows.
            window.pairApi.cluster.onInviteReceived(invite => {
                if (invite.state === 'pending') set({ activeInviteId: invite.inviteId })
            }),
            window.pairApi.cluster.onPendingInvitesChanged(invites => {
                set(state => ({
                    pendingInvites: pendingWithResults(invites, state.inviteResults),
                    activeInviteId:
                        state.activeInviteId && state.inviteResults[state.activeInviteId]
                            ? state.activeInviteId
                            : stillActive(state.activeInviteId, invites)
                }))
                // Pruning says this invite left the pending set, but not why.
                // Fetch its exact outcome even when the approval modal is closed.
                for (const result of Object.values(get().inviteResults)) {
                    if (
                        result.state === 'pending' &&
                        !invites.some(invite => invite.inviteId === result.inviteId)
                    )
                        void get().pollInvite(result.inviteId)
                }
            }),
            window.pairApi.nodes.onMembersChanged(members => {
                receiveMembers(members)
            })
        )
        if (transport)
            unsubs.push(
                transport.onDisconnect(() => {
                    connected = false
                    invalidatePermissions(true)
                }),
                transport.onConnect(() => {
                    connected = true
                    invalidatePermissions(true)
                    void get().refresh()
                })
            )
        if (initial) get().hydrate(initial)
        else {
            await get().refresh()
            return
        }
        if (connected) await get().refreshRemoteManagement()
    },

    refresh: async () => {
        const currentGeneration = permissionGeneration
        try {
            const snapshot = await window.pairApi.cluster.getInitial()
            if (currentGeneration !== permissionGeneration || !connected) return
            get().hydrate(snapshot)
            await get().refreshRemoteManagement()
        } catch {
            if (currentGeneration !== permissionGeneration) return
            invalidatePermissions()
            set({ pendingInvites: [], members: [], activeInviteId: null })
        }
    },

    refreshRemoteManagement: () => {
        if (!connected) return Promise.resolve()
        const currentGeneration = permissionGeneration
        if (permissionRead?.generation === currentGeneration) return permissionRead.request
        set({ remoteManagementLoading: true })
        const request = Promise.resolve()
            .then(() =>
                currentGeneration === permissionGeneration && connected
                    ? window.pairApi.cluster.getRemoteManagement()
                    : null
            )
            .then(snapshot => {
                if (!snapshot || currentGeneration !== permissionGeneration || !connected) return
                receivePermissions(snapshot, true)
            })
            .catch(error => {
                if (currentGeneration !== permissionGeneration) return
                invalidatePermissions()
                set({ remoteManagementError: getErrorString(error) })
            })
            .finally(() => {
                if (permissionRead?.request === request) permissionRead = null
                if (currentGeneration === permissionGeneration)
                    set({ remoteManagementLoading: false })
            })
        permissionRead = { generation: currentGeneration, request }
        return request
    },

    setRemoteManagement: async (peerUuid, allowed) => {
        const snapshot = get().remoteManagement
        if (
            !connected ||
            !snapshot?.revision ||
            !get().members.some(
                member => member.nodeUuid === peerUuid && member.state === 'member'
            ) ||
            !snapshot.grants.some(
                grant => grant.peerUuid === peerUuid && grant.state !== 'unavailable'
            ) ||
            get().remoteManagementChanges[peerUuid]?.phase === 'saving'
        )
            return
        const currentGeneration = permissionGeneration
        const change = (phase: RemoteManagementChange['phase'], message?: string) => {
            set(state => ({
                remoteManagementChanges: {
                    ...state.remoteManagementChanges,
                    [peerUuid]: { allowed, phase, message }
                }
            }))
        }
        change('saving')
        try {
            const saved = await window.pairApi.cluster.setRemoteManagement({
                peerUuid,
                allowed,
                expectedRevision: snapshot.revision
            })
            if (currentGeneration !== permissionGeneration) return
            receivePermissions(saved)
            if (!saved.revision) {
                change(
                    'unknown',
                    'The permission save could not be confirmed. Read back before trying again.'
                )
                await get().refreshRemoteManagement()
                return
            }
            const actual = get().remoteManagement?.grants.find(grant => grant.peerUuid === peerUuid)
            change(
                actual?.allowed === allowed && actual.state === (allowed ? 'allowed' : 'denied')
                    ? 'saved'
                    : 'conflict'
            )
        } catch (error) {
            if (currentGeneration !== permissionGeneration) return
            const code = serviceErrorCode(error)
            change(
                code === POLICY_CONFLICT_CODE
                    ? 'conflict'
                    : code === INVALID_PARAMS_CODE
                      ? 'failed'
                      : 'unknown',
                getErrorString(error)
            )
            await get().refreshRemoteManagement()
        }
    },

    cleanup: () => {
        generation++
        invalidatePermissions(true)
        permissionNeedsBaseline = false
        localIdentity = ''
        for (const inviteId of invitePolls.keys()) stopPolling(inviteId)
        unsubs.forEach(u => u())
        unsubs = []
        set({
            pendingInvites: [],
            members: [],
            activeInviteId: null,
            inviteResults: {},
            remoteManagementChanges: {},
            remoteManagementContext: null,
            remoteManagementError: null
        })
    },

    setActiveInvite: inviteId => {
        set({ activeInviteId: inviteId })
    },

    clearActiveInvite: () => {
        set({ activeInviteId: null })
    },

    respondToInvite: async (inviteId, ...response) => {
        const accept = response[0]
        const previous = get().inviteResults[inviteId]
        if (previous) return previous
        const invite = get().pendingInvites.find(value => value.inviteId === inviteId)
        const currentGeneration = generation
        if (accept && invite) {
            set(state => ({
                inviteResults: { ...state.inviteResults, [inviteId]: invite }
            }))
            ensurePolling(inviteId)
        }
        try {
            const result = await window.pairApi.cluster.respondToInvite(inviteId, ...response)
            if (generation === currentGeneration && accept && result.inviteId === inviteId)
                applyResult(result)
            return get().inviteResults[inviteId] ?? result
        } catch (error) {
            if (
                generation === currentGeneration &&
                invite &&
                accept &&
                isInviteStatusUnknown(getErrorString(error))
            )
                applyResult({
                    ...invite,
                    state: 'expired',
                    reason: 'outcome-unknown'
                })
            throw error
        }
    },

    pollInvite: async inviteId => {
        const invite = get().inviteResults[inviteId]
        if (!invite || invite.state !== 'pending') return
        const poll = ensurePolling(inviteId)
        if (poll.request) return poll.request
        const currentGeneration = generation
        const current = () => generation === currentGeneration && invitePolls.get(inviteId) === poll
        poll.request = Promise.resolve()
            .then(() => window.pairApi.cluster.inviteStatus(inviteId))
            .then(result => {
                if (current() && result.inviteId === inviteId) applyResult(result)
            })
            .catch(error => {
                if (current() && isInviteStatusUnknown(getErrorString(error)))
                    applyResult({ ...invite, state: 'expired', reason: 'outcome-unknown' })
                // Ordinary transport errors retain pending; the store-owned cadence retries.
            })
            .finally(() => {
                if (current()) poll.request = null
            })
        return poll.request
    }
}))
