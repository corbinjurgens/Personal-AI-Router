// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { APP_DISPLAY_NAME } from '@/shared/constants/app'
import type { Invite, JoinRequest } from '@/shared/types/cluster'

export function formatJoinRequestError(message: string): string {
    if (message.includes('already-clustered'))
        return 'This device already has cluster peers. Leave its current cluster before asking to join another.'
    if (message.includes('approver-changed'))
        return 'The selected member changed clusters. Ask a current member to review a new request.'
    if (message.includes('unsupported'))
        return 'Update PAIR on both devices to ask to join a cluster.'
    if (message.includes('request-status-lost'))
        return 'The request status is no longer available. Joining could not be confirmed. Check both devices.'
    return `${message} Joining could not be confirmed. Read the request status before trying again.`
}

export function joinRequestMessage(request: JoinRequest): string {
    switch (request.reason) {
        case 'unsupported-peer':
            return 'The selected device does not support Ask to join. Update PAIR on that device.'
        case 'already-clustered':
            return 'This device already has cluster peers. Leave its current cluster before asking to join another.'
        case 'requester-changed':
            return 'The requesting device changed while joining. Joining was not confirmed. Read its current cluster status before trying again.'
        case 'approver-changed':
            return 'The selected member changed clusters. Ask a current member to review a new request.'
        case 'request-status-lost':
        case 'outcome-unknown':
            return 'The request status is no longer available. Joining could not be confirmed. Check the cluster on both devices before trying again.'
        case 'awaiting-confirmation':
            return 'Waiting for pairing confirmation.'
        case 'rollback-pending':
            return 'Pairing could not complete. Waiting for cleanup; joining is not confirmed.'
    }
    if (isInviteConnectionFailure(request.reason) || request.reason === 'connection-failed')
        return "The join connection failed. Check the address, pairing port, and that PAIR is running on both devices. The member must also reach the requesting device's pairing port. Check firewalls on both devices and local-network reachability."
    switch (request.state) {
        case 'awaiting-approval':
            return `Waiting for approval on ${request.peerName || request.peerAddress}`
        case 'awaiting-requester':
            return 'Preparing to pair'
        case 'pairing':
            return request.direction === 'incoming'
                ? 'Preparing to pair. Read the PIN displayed on this member and enter it on the requesting device.'
                : 'Preparing to pair. Enter the PIN from the selected member in the pairing prompt.'
        case 'paired':
            return request.direction === 'incoming'
                ? 'The requesting device was admitted to this cluster.'
                : 'Joined the cluster.'
        case 'declined':
            return 'The join request was declined.'
        case 'canceled':
            return 'Join request canceled.'
        case 'expired':
            return 'The join request expired. Joining was not confirmed.'
        default:
            return `Joining was not confirmed${request.reason ? `: ${request.reason}` : '.'}`
    }
}

export const REVERSE_INVITE_GUIDANCE =
    'If the other device blocks incoming connections, open PAIR on that device and invite this computer instead.'

export function isInviteConnectionFailure(reason: string): boolean {
    return [
        'connection-reset',
        'connection-refused',
        'connection-timeout',
        'connection-unreachable'
    ].includes(reason)
}

export function inviteFailureMessage(invite: Invite, receiver = false): string {
    if (invite.reason === 'incorrect-pin') {
        return receiver ? INCORRECT_PIN_RECEIVER_MESSAGE : INCORRECT_PIN_SENDER_MESSAGE
    }
    if (invite.error) return invite.error
    switch (invite.reason) {
        case 'connection-reset':
            return 'The connection to the other node was reset.'
        case 'connection-refused':
            return 'The other node refused the connection.'
        case 'connection-timeout':
            return 'The connection to the other node timed out.'
        case 'connection-unreachable':
            return 'The other node could not be reached.'
        default:
            return invite.reason
                ? `Pairing failed: ${invite.reason}`
                : 'Pairing failed. Please try again.'
    }
}

export function isInviteStatusUnknown(message: string): boolean {
    return (
        message.includes('-32001') ||
        message.toLowerCase().includes('unknown invite') ||
        message.toLowerCase().includes('invite session evicted')
    )
}

const DEFAULT_INVITE_ERROR = `Could not invite that node. Check the IP address, port, and that ${APP_DISPLAY_NAME} is running on the other machine, then try again.`

/**
 * Shown when a pairing session was torn down and can no longer be answered
 * (a wrong PIN evicts the backend session, `-32001`). The invite is terminal;
 * the user must restart the flow with a fresh invite.
 */
const INVITE_SESSION_ENDED_MESSAGE =
    'This invitation is no longer valid — ask the inviting node to send a new one.'

/** Wrong-PIN copy for the joiner (the node that entered the PIN). */
const INCORRECT_PIN_RECEIVER_MESSAGE = `Incorrect PIN. ${INVITE_SESSION_ENDED_MESSAGE}`

/** Wrong-PIN copy mirrored on the inviter (the node that issued the PIN). */
const INCORRECT_PIN_SENDER_MESSAGE =
    'Incorrect PIN entered on the other node. Send a new invite to try again.'

export function formatClusterInviteError(message: string | null | undefined): string {
    const normalizedMessage = message ?? DEFAULT_INVITE_ERROR
    const normalized = normalizedMessage.toLowerCase()

    // A wrong PIN evicts the backend pairing session, so a stale retry surfaces
    // `-32001: invite session evicted`. Normalize it (and any raw wrong-PIN
    // text) to the terminal "no longer valid" copy instead of leaking the code.
    if (
        normalized.includes('-32001') ||
        normalized.includes('invite session evicted') ||
        normalized.includes('incorrect pin')
    ) {
        return INVITE_SESSION_ENDED_MESSAGE
    }

    if (normalized.includes('deadline_exceeded') || normalized.includes('deadline exceeded')) {
        return `Could not reach that node before the connection timed out. Check the IP address, port, and that ${APP_DISPLAY_NAME} is running on the other machine, then try again.`
    }

    if (
        normalized.includes('unavailable') ||
        normalized.includes('econnrefused') ||
        normalized.includes('connection refused') ||
        normalized.includes('name resolution')
    ) {
        return 'Could not connect to that node. Check the IP address, join port, and that the other machine is online.'
    }

    return normalizedMessage
}
