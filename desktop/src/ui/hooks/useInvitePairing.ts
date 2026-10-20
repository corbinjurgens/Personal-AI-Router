// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Invite } from '@/shared/types/cluster'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'
import getErrorString from '@/shared/utils/get-error-string'
import {
    formatClusterInviteError,
    inviteFailureMessage,
    isInviteStatusUnknown
} from '@/ui/utils/cluster-invite-error'

interface InvitePairing {
    /** The outbound invite (carries the PIN to display) once started; null before. */
    invite: Invite | null
    /** True while the initial `cluster:invite-node` request is in flight. */
    submitting: boolean
    error: string | null
    allowRemoteManagement: boolean
    setAllowRemoteManagement: (allowed: boolean) => void
    submittedPermission: boolean | null
    /** Begin PIN pairing with a node, then poll its status until it resolves. */
    start: (ipAddress: string) => Promise<void>
    /**
     * Cancel a still-pending outbound invite: tell the backend to tear down the
     * pairing session (invalidating the PIN so a remote user can no longer
     * complete the join), dissolve any throwaway solo cluster, then clear local
     * state. Falls back to a local reset when there is no live invite.
     */
    cancel: () => Promise<void>
    /** Clear the invite and stop polling (local only; does not tell the backend). */
    reset: () => void
}

/**
 * Drives an outbound PIN-pairing session: fires `cluster:invite-node`, surfaces
 * the returned PIN, and polls `cluster:invite-status` until the invite leaves the
 * `pending` state (paired / declined / expired / failed).
 */
export function useInvitePairing(): InvitePairing {
    const [invite, setInvite] = useState<Invite | null>(null)
    const [submitting, setSubmitting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [allowRemoteManagement, setAllowRemoteManagement] = useState(false)
    const [submittedPermission, setSubmittedPermission] = useState<boolean | null>(null)
    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const inviteIdRef = useRef<string | null>(null)
    const generationRef = useRef(0)

    const stopPolling = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
        }
    }, [])

    const reset = useCallback(() => {
        generationRef.current++
        stopPolling()
        inviteIdRef.current = null
        setInvite(null)
        setSubmitting(false)
        setError(null)
        setAllowRemoteManagement(false)
        setSubmittedPermission(null)
    }, [stopPolling])

    useEffect(
        () => () => {
            generationRef.current++
            stopPolling()
        },
        [stopPolling]
    )

    const cancel = useCallback(async () => {
        const id = inviteIdRef.current
        const generation = generationRef.current
        stopPolling()
        if (id) {
            try {
                await window.pairApi.cluster.cancelInvite(id)
            } catch {
                // Best-effort: the backend's session TTL and the client sweep are
                // the backstops if this fails (e.g. it already resolved).
            }
            if (generation !== generationRef.current || inviteIdRef.current !== id) return
            // The invite may have auto-created a throwaway solo cluster; drop it
            // if no peer joined (same as a terminal non-paired outcome).
            void window.pairApi.cluster.abandonIfSolo()
        }
        reset()
    }, [reset, stopPolling])

    const start = useCallback(
        async (ipAddress: string) => {
            const generation = ++generationRef.current
            inviteIdRef.current = null
            setInvite(null)
            setSubmitting(true)
            setError(null)
            setSubmittedPermission(allowRemoteManagement)
            stopPolling()
            try {
                const result = await window.pairApi.cluster.inviteNode(
                    ipAddress,
                    allowRemoteManagement
                )
                if (generation !== generationRef.current) return
                setInvite(result)
                if (result.state === 'pending' && result.inviteId) {
                    inviteIdRef.current = result.inviteId
                    let polling = false
                    pollRef.current = setInterval(() => {
                        const id = inviteIdRef.current
                        if (!id || polling) return
                        polling = true
                        void window.pairApi.cluster
                            .inviteStatus(id)
                            .then(status => {
                                if (generation !== generationRef.current || status.inviteId !== id)
                                    return
                                setInvite(status)
                                if (status.state !== 'pending') {
                                    stopPolling()
                                    // A pairing that never completes leaves behind a
                                    // cluster that may have been auto-created just for
                                    // this invite; dissolve it if we are still solo.
                                    if (status.state !== 'paired') {
                                        void window.pairApi.cluster.abandonIfSolo()
                                    }
                                }
                            })
                            .catch(err => {
                                if (generation !== generationRef.current) return
                                if (isInviteStatusUnknown(getErrorString(err))) {
                                    setInvite({
                                        ...result,
                                        state: 'expired',
                                        reason: 'outcome-unknown'
                                    })
                                    stopPolling()
                                }
                            })
                            .finally(() => {
                                polling = false
                            })
                    }, MODULAR_INVITE_STATUS_POLL_INTERVAL_MS)
                } else if (result.state !== 'paired') {
                    // Any non-paired terminal initial result (failed / rejected /
                    // declined / expired). `rejected` (e.g. the target is already
                    // clustered, `reason: 'already-clustered'`) matched neither
                    // branch before, so its solo cluster was never dissolved. The
                    // panel renders the state/reason-specific copy; a bare `failed`
                    // also surfaces the generic error banner.
                    if (result.state === 'failed') setError(inviteFailureMessage(result))
                    void window.pairApi.cluster.abandonIfSolo()
                }
            } catch (err) {
                if (generation !== generationRef.current) return
                setError(formatClusterInviteError(getErrorString(err)))
                void window.pairApi.cluster.abandonIfSolo()
            } finally {
                if (generation === generationRef.current) setSubmitting(false)
            }
        },
        [stopPolling, allowRemoteManagement]
    )

    return {
        invite,
        submitting,
        error,
        start,
        cancel,
        reset,
        allowRemoteManagement,
        setAllowRemoteManagement,
        submittedPermission
    }
}
