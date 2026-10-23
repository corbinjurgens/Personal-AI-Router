// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useId, useState } from 'react'
import {
    Button,
    Flex,
    ModalContent,
    ModalDialog,
    ModalRoot,
    Stack,
    Text
} from '@nvidia/foundations-react-core'
import type { Invite, JoinRequest } from '@/shared/types/cluster'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'
import getErrorString from '@/shared/utils/get-error-string'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { InvitePairingPanel, RemoteManagementChoice } from '@/ui/components/InvitePairingPanel'
import { DialogHeader } from '@/ui/components/DialogHeader'
import { isInviteStatusUnknown, joinRequestMessage } from '@/ui/utils/cluster-invite-error'

export function SoloReplacementConfirmation({
    open,
    onConfirm,
    onDismiss
}: {
    open: boolean
    onConfirm: () => void
    onDismiss: () => void
}) {
    const headingId = useId()
    return (
        <ModalRoot
            open={open}
            onOpenChange={next => {
                if (!next) onDismiss()
            }}
            hideCloseButton
        >
            <ModalDialog aria-labelledby={headingId}>
                <ModalContent>
                    <DialogHeader onClose={onDismiss}>
                        <span id={headingId}>Ask to join</span>
                    </DialogHeader>
                    <Stack gap="3">
                        <Text>
                            Joining replaces this device&apos;s empty cluster. Other devices will
                            not be moved.
                        </Text>
                        <Text kind="body/regular/sm">
                            Replacement happens after approval. If pairing then fails or is
                            canceled, this device remains unclustered.
                        </Text>
                        <Flex justify="end" gap="2">
                            <Button kind="secondary" onClick={onDismiss}>
                                Cancel
                            </Button>
                            <Button color="brand" onClick={onConfirm}>
                                Ask to join
                            </Button>
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}

export default function JoinRequestCard({ request }: { request: JoinRequest }) {
    const [allowed, setAllowed] = useState(false)
    const [invite, setInvite] = useState<Invite | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [statusRead, setStatusRead] = useState(0)
    const busy = useClusterInvitationsStore(state => state.joinBusy[request.requestId] === true)
    const permissions = useClusterInvitationsStore(state => state.remoteManagement)
    const savedPermission = permissions?.grants.find(
        grant => grant.peerUuid === request.peerNodeUuid
    )?.allowed
    const active = ['awaiting-approval', 'awaiting-requester', 'pairing'].includes(request.state)
    const incoming = request.direction === 'incoming'
    useEffect(() => {
        setInvite(null)
        setError(null)
        if (!incoming || !request.inviteId) return
        let disposed = false
        let reading = false
        let settled = false
        const id = request.inviteId
        const read = async () => {
            if (reading || settled) return
            reading = true
            try {
                const result = await window.pairApi.cluster.inviteStatus(id)
                if (!disposed && result.inviteId === id) {
                    setInvite(previous =>
                        previous && previous.state !== 'pending' ? previous : result
                    )
                    settled = result.state !== 'pending'
                    setError(null)
                }
            } catch (failure) {
                if (!disposed) {
                    settled = isInviteStatusUnknown(getErrorString(failure))
                    setInvite(null)
                    setError(
                        `Pairing status unavailable: ${getErrorString(failure)}. Joining is not confirmed; check both devices.`
                    )
                }
            } finally {
                reading = false
            }
        }
        void read()
        const timer = setInterval(() => void read(), MODULAR_INVITE_STATUS_POLL_INTERVAL_MS)
        return () => {
            disposed = true
            clearInterval(timer)
        }
    }, [incoming, request.inviteId, statusRead])
    const cancel = () =>
        void useClusterInvitationsStore.getState().cancelJoinRequest(request.requestId)
    const readStatus = () => {
        setStatusRead(value => value + 1)
        void useClusterInvitationsStore.getState().refreshJoinRequests()
    }
    return (
        <div className="settings-card node-item-card pair-paper p-4">
            <Stack gap="3">
                <Text kind="body/semibold/sm">{request.peerName || request.peerAddress}</Text>
                <Text kind="body/regular/sm">
                    Unverified claimed device: {request.peerAddress}:{request.peerPort}
                </Text>
                <Text kind="body/regular/sm">
                    Expires {new Date(request.expiresAt).toLocaleString()}
                </Text>
                <Text kind="body/regular/sm" role="status">
                    {incoming && request.state === 'awaiting-approval'
                        ? 'This device is asking to join your cluster. Review this unverified request before approving.'
                        : joinRequestMessage(request)}
                </Text>
                {!incoming && request.state === 'awaiting-approval' && (
                    <Text kind="body/regular/sm">
                        Open PAIR on that device, then Settings &gt; Cluster, to review this
                        request.
                    </Text>
                )}
                {!incoming && request.state === 'pairing' && request.inviteId && (
                    <Button
                        color="brand"
                        disabled={busy}
                        onClick={() =>
                            void useClusterInvitationsStore
                                .getState()
                                .recoverJoinInvite(request.requestId, true)
                        }
                    >
                        Enter PIN
                    </Button>
                )}
                {incoming && request.state === 'awaiting-approval' && (
                    <>
                        <RemoteManagementChoice
                            allowed={allowed}
                            onChange={setAllowed}
                            disabled={busy}
                        />
                        <Flex gap="2" justify="end">
                            <Button
                                kind="secondary"
                                disabled={busy}
                                onClick={() =>
                                    void useClusterInvitationsStore
                                        .getState()
                                        .respondToJoinRequest(request.requestId, false)
                                }
                            >
                                Decline
                            </Button>
                            <Button
                                color="brand"
                                disabled={busy}
                                onClick={() =>
                                    void useClusterInvitationsStore
                                        .getState()
                                        .respondToJoinRequest(request.requestId, true, allowed)
                                }
                            >
                                Approve
                            </Button>
                        </Flex>
                    </>
                )}
                {incoming && request.inviteId && (active || request.state === 'paired') && (
                    <InvitePairingPanel
                        invite={invite}
                        error={error}
                        onReset={readStatus}
                        onCancel={cancel}
                        savedPermission={savedPermission}
                    />
                )}
                {active && (
                    <Button kind="secondary" disabled={busy} onClick={cancel}>
                        Cancel request
                    </Button>
                )}
            </Stack>
        </div>
    )
}
