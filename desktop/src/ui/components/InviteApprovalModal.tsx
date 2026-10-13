// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react'
import {
    Button,
    Flex,
    ModalContent,
    ModalDialog,
    ModalRoot,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { DialogHeader } from './DialogHeader'
import { InlineErrorBanner } from './InlineErrorBanner'
import { useBlurOnOpen } from '@/ui/hooks/useBlurOnOpen'
import getErrorString from '@/shared/utils/get-error-string'
import { formatClusterInviteError, inviteFailureMessage } from '@/ui/utils/cluster-invite-error'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

export function InviteApprovalModal() {
    const activeInviteId = useClusterInvitationsStore(state => state.activeInviteId)
    const pendingInvites = useClusterInvitationsStore(state => state.pendingInvites)
    const results = useClusterInvitationsStore(state => state.inviteResults)
    const respondToInvite = useClusterInvitationsStore(state => state.respondToInvite)
    const pollInvite = useClusterInvitationsStore(state => state.pollInvite)
    const clearActiveInvite = useClusterInvitationsStore(state => state.clearActiveInvite)
    const shown = activeInviteId
        ? (results[activeInviteId] ??
          pendingInvites.find(invite => invite.inviteId === activeInviteId))
        : null
    const submitted = Boolean(activeInviteId && results[activeInviteId])
    const terminal = Boolean(shown && shown.state !== 'pending')
    const open = Boolean(shown && shown.state !== 'paired')
    const inputRef = useRef<HTMLInputElement>(null)
    const [pin, setPin] = useState('')
    const [error, setError] = useState<string | null>(null)
    useBlurOnOpen(open)

    useEffect(() => {
        setPin('')
        setError(null)
    }, [activeInviteId])
    useEffect(() => {
        if (shown?.state === 'paired') clearActiveInvite()
    }, [shown?.state, clearActiveInvite])

    useEffect(() => {
        if (!activeInviteId || !submitted || terminal) return
        // The store owns retry cadence across modal dismissal and card reopening.
        void pollInvite(activeInviteId)
    }, [activeInviteId, submitted, terminal, pollInvite])

    useEffect(() => {
        if (!open || submitted) return
        const id = requestAnimationFrame(() => inputRef.current?.focus())
        return () => cancelAnimationFrame(id)
    }, [open, submitted])

    const handleAccept = useCallback(async () => {
        if (!shown || submitted) return
        if (pin.length !== 6) {
            setError('Enter the 6-digit PIN shown on the inviting node.')
            return
        }
        const id = shown.inviteId
        try {
            await respondToInvite(id, true, pin)
        } catch (err) {
            if (useClusterInvitationsStore.getState().activeInviteId === id)
                setError(formatClusterInviteError(getErrorString(err)))
        }
    }, [shown, submitted, pin, respondToInvite])

    const handleDismiss = useCallback(async () => {
        if (!shown || submitted || terminal) {
            clearActiveInvite()
            return
        }
        try {
            await respondToInvite(shown.inviteId, false)
        } catch {
            /* service retains the outcome */
        }
        if (useClusterInvitationsStore.getState().activeInviteId === shown.inviteId)
            clearActiveInvite()
    }, [shown, submitted, terminal, respondToInvite, clearActiveInvite])

    if (!shown) return null
    const message =
        shown.state === 'failed'
            ? inviteFailureMessage(shown, true)
            : terminal
              ? shown.reason === 'outcome-unknown'
                  ? 'The invitation status is no longer available. Pairing could not be confirmed. Check the cluster on both devices.'
                  : `Invitation ${shown.state}. Pairing was not confirmed; check the cluster on both devices.`
              : null
    const pendingMessage =
        shown.reason === 'rollback-pending'
            ? 'Pairing could not complete. Waiting for cleanup to finish...'
            : 'Waiting for pairing confirmation...'

    return (
        <ModalRoot open={open} onOpenChange={next => !next && void handleDismiss()} hideCloseButton>
            <ModalDialog
                closeOnClickOutside={false}
                onEscapeKeyDown={event => event.preventDefault()}
            >
                <ModalContent className="no-drag-elements max-content-modal">
                    <DialogHeader onClose={() => void handleDismiss()}>
                        Cluster invitation
                    </DialogHeader>
                    <Stack gap="4" className="pt-2">
                        {(message || error) && <InlineErrorBanner message={message ?? error} />}
                        <Text kind="body/regular/sm">You have been invited to join a cluster:</Text>
                        <Flex align="center" gap="2">
                            <Text kind="body/semibold/sm" className="min-w-15">
                                From
                            </Text>
                            <Text kind="body/regular/sm">
                                {shown.fromNodeName || shown.fromNodeId}
                            </Text>
                        </Flex>
                        {!terminal && (
                            <Stack gap="1">
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    {submitted
                                        ? pendingMessage
                                        : `Enter the 6-digit PIN shown on ${shown.fromNodeName || 'the inviting node'}.`}
                                </Text>
                                <TextInput
                                    ref={inputRef}
                                    value={pin}
                                    onValueChange={value =>
                                        setPin(value.replace(/\D/g, '').slice(0, 6))
                                    }
                                    placeholder="000000"
                                    inputMode="numeric"
                                    maxLength={6}
                                    disabled={submitted}
                                    onKeyDown={event => {
                                        if (event.key === 'Enter') void handleAccept()
                                    }}
                                    style={{ fontFamily: 'monospace', letterSpacing: '0.3em' }}
                                />
                            </Stack>
                        )}
                        <Flex justify="end" gap="2">
                            {terminal || submitted ? (
                                <Button kind="secondary" size="small" onClick={clearActiveInvite}>
                                    Close
                                </Button>
                            ) : (
                                <>
                                    <Button
                                        kind="secondary"
                                        size="small"
                                        onClick={() => void handleDismiss()}
                                    >
                                        Decline
                                    </Button>
                                    <Button
                                        kind="primary"
                                        size="small"
                                        color="brand"
                                        onClick={() => void handleAccept()}
                                        disabled={pin.length !== 6}
                                    >
                                        Join cluster
                                    </Button>
                                </>
                            )}
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
