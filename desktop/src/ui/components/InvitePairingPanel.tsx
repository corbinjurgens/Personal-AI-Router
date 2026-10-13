// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { InlineErrorBanner } from './InlineErrorBanner'
import {
    inviteFailureMessage,
    isInviteConnectionFailure,
    REVERSE_INVITE_GUIDANCE
} from '@/ui/utils/cluster-invite-error'
import type { Invite } from '@/shared/types/cluster'

interface InvitePairingPanelProps {
    invite: Invite | null
    error: string | null
    onReset: () => void
    /**
     * Cancel a still-pending invite. Unlike `onReset` (local-only), this tears
     * down the backend pairing session and invalidates the PIN. Defaults to
     * `onReset` when not supplied.
     */
    onCancel?: () => void
    /** Called from the success state; defaults to `onReset`. */
    onDone?: () => void
}

/**
 * Renders the inviter side of a PIN-pairing session: the 6-digit PIN to read out
 * to the other node, the waiting/terminal states, and a retry affordance.
 */
export function InvitePairingPanel({
    invite,
    error,
    onReset,
    onCancel,
    onDone
}: InvitePairingPanelProps) {
    if (error && !invite) {
        return (
            <Stack gap="2">
                <InlineErrorBanner message={error} onClose={onReset} />
                <Flex justify="end">
                    <Button kind="secondary" size="small" onClick={onReset}>
                        Try again
                    </Button>
                </Flex>
            </Stack>
        )
    }

    if (!invite) return null

    if (invite.state === 'pending') {
        const confirming =
            invite.reason === 'awaiting-confirmation' || invite.reason === 'rollback-pending'
        return (
            <Stack gap="3" align="center">
                {!confirming && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Enter this PIN on the other device to confirm pairing:
                    </Text>
                )}
                {!confirming && (
                    <Text
                        kind="title/md"
                        style={{ fontFamily: 'monospace', letterSpacing: '0.3em' }}
                    >
                        {invite.pin ?? '------'}
                    </Text>
                )}
                <Flex align="center" gap="2">
                    <span className="spinner-element" role="status" aria-label="" />
                    <Text kind="body/regular/sm">
                        {invite.reason === 'rollback-pending'
                            ? 'Pairing could not complete. Waiting for cleanup to finish...'
                            : confirming
                              ? 'Waiting for pairing confirmation...'
                              : 'Waiting for the other node to confirm...'}
                    </Text>
                </Flex>
                <Button kind="secondary" size="small" onClick={onCancel ?? onReset}>
                    Cancel
                </Button>
            </Stack>
        )
    }

    if (invite.state === 'paired') {
        return (
            <Stack gap="3" align="center">
                <Text kind="body/semibold/md">Node paired successfully.</Text>
                <Button kind="primary" color="brand" size="small" onClick={onDone ?? onReset}>
                    Done
                </Button>
            </Stack>
        )
    }

    // Failed invites retain their diagnostic and offer a fresh invitation.
    if (invite.state === 'failed') {
        return (
            <Stack gap="2">
                <InlineErrorBanner
                    message={error ?? inviteFailureMessage(invite)}
                    onClose={onReset}
                />
                {isInviteConnectionFailure(invite.reason) && (
                    <Text kind="body/regular/sm">{REVERSE_INVITE_GUIDANCE}</Text>
                )}
                <Flex justify="end">
                    <Button kind="secondary" size="small" onClick={onReset}>
                        Try again
                    </Button>
                </Flex>
            </Stack>
        )
    }

    const message =
        invite.state === 'declined'
            ? 'The other node declined the invite.'
            : invite.state === 'canceled'
              ? 'Invite canceled.'
              : invite.state === 'rejected'
                ? invite.reason === 'already-clustered'
                    ? 'That node is already in a cluster.'
                    : 'The other node refused the invite.'
                : invite.state === 'expired'
                  ? invite.reason === 'outcome-unknown'
                      ? 'The invitation status is no longer available. Pairing could not be confirmed. Check the cluster on both devices.'
                      : 'The invite expired before it was confirmed. Check the cluster on both devices.'
                  : 'Pairing failed. Please try again.'

    return (
        <Stack gap="3" align="center">
            <Text kind="body/regular/sm">{message}</Text>
            <Button kind="secondary" size="small" onClick={onReset}>
                Try again
            </Button>
        </Stack>
    )
}
