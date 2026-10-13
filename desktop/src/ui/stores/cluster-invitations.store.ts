// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ClusterNode, Invite } from '@/shared/types/cluster'
import getErrorString from '@/shared/utils/get-error-string'
import { isInviteStatusUnknown } from '@/ui/utils/cluster-invite-error'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'

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
    initialize: (initial?: ClusterInitialSnapshot) => Promise<void>
    hydrate: (snapshot: ClusterInitialSnapshot) => void
    refresh: () => Promise<void>
    cleanup: () => void
    /** Surface a specific pending invite in the approval modal (e.g. from the card). */
    setActiveInvite: (inviteId: string) => void
    /** Close the approval modal without responding. */
    clearActiveInvite: () => void
    /** Accept (with PIN) or decline a specific inbound invite. */
    respondToInvite: (inviteId: string, accept: boolean, pin?: string) => Promise<Invite>
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

    hydrate: snapshot => {
        set(state => ({
            pendingInvites: pendingWithResults(snapshot.pendingInvites, state.inviteResults),
            members: snapshot.members,
            activeInviteId:
                state.activeInviteId && state.inviteResults[state.activeInviteId]
                    ? state.activeInviteId
                    : stillActive(state.activeInviteId, snapshot.pendingInvites)
        }))
    },

    initialize: async initial => {
        if (initial) {
            get().hydrate(initial)
        } else {
            await get().refresh()
        }

        if (!window.pairApi) return
        unsubs.push(
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
                set({ members })
            })
        )
    },

    refresh: async () => {
        try {
            const snapshot = await window.pairApi.cluster.getInitial()
            get().hydrate(snapshot)
        } catch {
            set({ pendingInvites: [], members: [], activeInviteId: null })
        }
    },

    cleanup: () => {
        generation++
        for (const inviteId of invitePolls.keys()) stopPolling(inviteId)
        unsubs.forEach(u => u())
        unsubs = []
        set({
            pendingInvites: [],
            members: [],
            activeInviteId: null,
            inviteResults: {}
        })
    },

    setActiveInvite: inviteId => {
        set({ activeInviteId: inviteId })
    },

    clearActiveInvite: () => {
        set({ activeInviteId: null })
    },

    respondToInvite: async (inviteId, accept, pin) => {
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
            const result = await window.pairApi.cluster.respondToInvite(inviteId, accept, pin)
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
