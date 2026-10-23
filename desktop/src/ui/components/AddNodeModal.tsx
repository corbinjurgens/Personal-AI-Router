// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react'
import {
    Button,
    Divider,
    Flex,
    FormField,
    ModalContent,
    ModalDialog,
    ModalRoot,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { DialogHeader } from './DialogHeader'
import { InvitePairingPanel, RemoteManagementChoice } from './InvitePairingPanel'
import { useBlurOnOpen } from '@/ui/hooks/useBlurOnOpen'
import { useInvitePairing } from '@/ui/hooks/useInvitePairing'
import { useInvitablePeers } from '@/ui/hooks/useInvitablePeers'
import { useAskToJoin } from '@/ui/hooks/useAskToJoin'
import JoinRequestCard, {
    SoloReplacementConfirmation
} from '@/ui/components/ClusterSettings/JoinRequestCard'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

interface AddNodeModalProps {
    open: boolean
    onOpenChange: (open: boolean) => void
}

export function AddNodeModal({ open, onOpenChange }: AddNodeModalProps) {
    useBlurOnOpen(open)
    const [manualIp, setManualIp] = useState('')
    const pairing = useInvitePairing()
    const joining = useAskToJoin()
    const dismissConfirmation = joining.dismiss
    const joinSnapshot = useClusterInvitationsStore(state => state.joinSnapshot)
    const resetPairing = pairing.reset
    const nodesThatCanBeAdded = useInvitablePeers()
    useEffect(() => {
        if (!open) {
            resetPairing()
            dismissConfirmation()
            setManualIp('')
        }
    }, [open, resetPairing, dismissConfirmation])

    const handleOpenChange = useCallback(
        (next: boolean) => {
            setManualIp('')
            pairing.reset()
            joining.dismiss()
            onOpenChange(next)
        },
        [onOpenChange, pairing, joining]
    )

    const handleManualInvite = useCallback(() => {
        const ip = manualIp.trim()
        if (
            !ip ||
            pairing.submitting ||
            pairing.invite?.state === 'pending' ||
            useClusterInvitationsStore.getState().joinBusy.outgoing ||
            joinSnapshot?.requests.some(
                request =>
                    request.direction === 'outgoing' &&
                    ['awaiting-approval', 'awaiting-requester', 'pairing'].includes(request.state)
            )
        )
            return
        void pairing.start(ip)
    }, [manualIp, pairing, joinSnapshot])

    const showPairing = pairing.invite !== null || pairing.error !== null
    const outgoing =
        joinSnapshot?.requests.filter(request => request.direction === 'outgoing') ?? []
    const inviteInFlight =
        pairing.submitting ||
        pairing.invite?.state === 'pending' ||
        joining.submitting ||
        outgoing.some(request =>
            ['awaiting-approval', 'awaiting-requester', 'pairing'].includes(request.state)
        )

    return (
        <ModalRoot open={open} onOpenChange={handleOpenChange} hideCloseButton>
            <ModalDialog>
                <ModalContent className="no-drag-elements max-content-modal">
                    <DialogHeader onClose={() => handleOpenChange(false)}>
                        <Flex align="center" gap="2">
                            <span>Add node</span>
                            {pairing.submitting && (
                                <span className="spinner-element" role="status" aria-label="" />
                            )}
                        </Flex>
                    </DialogHeader>
                    <Stack gap="4" className="pt-2">
                        {outgoing.map(request => (
                            <JoinRequestCard key={request.requestId} request={request} />
                        ))}
                        {joining.error && <Text role="alert">{joining.error}</Text>}
                        {showPairing ? (
                            <InvitePairingPanel
                                invite={pairing.invite}
                                error={pairing.error}
                                onReset={pairing.reset}
                                onCancel={() => void pairing.cancel()}
                                onDone={() => handleOpenChange(false)}
                                savedPermission={pairing.savedPermission}
                            />
                        ) : (
                            <>
                                <RemoteManagementChoice
                                    allowed={pairing.allowRemoteManagement}
                                    onChange={pairing.setAllowRemoteManagement}
                                    disabled={inviteInFlight}
                                />
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    For Ask to join, choose this device&apos;s remote-management
                                    permission in the PIN prompt after approval.
                                </Text>
                                <Flex align="end" gap="2" wrap="wrap">
                                    <FormField slotLabel="IP address" className="flex-1">
                                        <TextInput
                                            value={manualIp}
                                            onValueChange={setManualIp}
                                            placeholder="192.168.1.100"
                                            onKeyDown={event => {
                                                if (event.key === 'Enter') handleManualInvite()
                                            }}
                                            disabled={inviteInFlight}
                                        />
                                    </FormField>
                                    <Button
                                        kind="primary"
                                        color="brand"
                                        onClick={handleManualInvite}
                                        disabled={!manualIp.trim() || inviteInFlight}
                                    >
                                        Invite
                                    </Button>
                                    <Button
                                        kind="secondary"
                                        onClick={() => joining.start(manualIp)}
                                        disabled={
                                            !manualIp.trim() || inviteInFlight || !!joining.blocked
                                        }
                                    >
                                        Ask to join
                                    </Button>
                                </Flex>
                                {joining.blocked && (
                                    <Text kind="body/regular/sm">{joining.blocked}</Text>
                                )}

                                {nodesThatCanBeAdded.length > 0 && (
                                    <Stack gap="4" className="mt-1">
                                        <Divider />

                                        <Stack gap="1">
                                            {nodesThatCanBeAdded.map(node => (
                                                <Flex
                                                    key={node.id}
                                                    align="center"
                                                    justify="between"
                                                    gap="2"
                                                    className="py-1"
                                                >
                                                    <Stack gap="0">
                                                        <Text
                                                            kind="body/semibold/sm"
                                                            className="uppercase"
                                                        >
                                                            {node.name || node.ipAddress}
                                                        </Text>
                                                        <Text
                                                            kind="body/regular/sm"
                                                            className="text-subtle-color"
                                                        >
                                                            {node.clustered
                                                                ? 'In another cluster'
                                                                : `${node.ipAddress}:${node.port}`}
                                                        </Text>
                                                    </Stack>
                                                    <Button
                                                        kind="primary"
                                                        color="brand"
                                                        size="small"
                                                        onClick={() =>
                                                            node.clustered
                                                                ? joining.start(
                                                                      node.ipAddress,
                                                                      node.port,
                                                                      node.id
                                                                  )
                                                                : void pairing.start(node.ipAddress)
                                                        }
                                                        disabled={
                                                            inviteInFlight ||
                                                            (node.clustered && !!joining.blocked)
                                                        }
                                                    >
                                                        {node.clustered ? 'Ask to join' : 'Invite'}
                                                    </Button>
                                                </Flex>
                                            ))}
                                        </Stack>
                                    </Stack>
                                )}
                            </>
                        )}
                    </Stack>
                    <SoloReplacementConfirmation
                        open={joining.confirmation !== null}
                        onConfirm={joining.confirm}
                        onDismiss={joining.dismiss}
                    />
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
