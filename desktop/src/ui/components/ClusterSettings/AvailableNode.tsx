// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { useEffect } from 'react'
import type { AvailableNode as AvailableNodeType } from '@/shared/types/cluster'
import { InvitePairingPanel, RemoteManagementChoice } from '@/ui/components/InvitePairingPanel'
import { useInvitePairing } from '@/ui/hooks/useInvitePairing'
import { useAskToJoin } from '@/ui/hooks/useAskToJoin'
import { SoloReplacementConfirmation } from './JoinRequestCard'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

export default function AvailableNode({ node }: { node: AvailableNodeType }) {
    const pairing = useInvitePairing()
    const joining = useAskToJoin()
    const joinSnapshot = useClusterInvitationsStore(state => state.joinSnapshot)
    const resetPairing = pairing.reset
    useEffect(() => resetPairing(), [node.id, resetPairing])
    const showPairing = pairing.invite !== null || pairing.error !== null
    const inviteInFlight =
        pairing.submitting ||
        pairing.invite?.state === 'pending' ||
        joining.submitting ||
        !!joinSnapshot?.requests.some(
            request =>
                request.direction === 'outgoing' &&
                ['awaiting-approval', 'awaiting-requester', 'pairing'].includes(request.state)
        )

    return (
        <div className="settings-card node-item-card pair-paper p-4">
            {showPairing ? (
                <InvitePairingPanel
                    invite={pairing.invite}
                    error={pairing.error}
                    onReset={pairing.reset}
                    onCancel={() => void pairing.cancel()}
                    savedPermission={pairing.savedPermission}
                />
            ) : (
                <Stack gap="3">
                    {!node.clustered && (
                        <RemoteManagementChoice
                            allowed={pairing.allowRemoteManagement}
                            onChange={pairing.setAllowRemoteManagement}
                            disabled={inviteInFlight}
                        />
                    )}
                    {node.clustered && joining.blocked && (
                        <Text kind="body/regular/sm">{joining.blocked}</Text>
                    )}
                    <Flex align="center" justify="between" gap="2">
                        <Stack gap="1">
                            <Text kind="body/semibold/sm">
                                {(node.name || node.ipAddress).toUpperCase()}
                            </Text>
                            <Text kind="body/regular/sm" className="text-subtle-color">
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
                                    ? joining.start(node.ipAddress, node.port, node.id)
                                    : void pairing.start(node.ipAddress)
                            }
                            disabled={inviteInFlight || (node.clustered && !!joining.blocked)}
                        >
                            {node.clustered ? 'Ask to join' : 'Invite'}
                        </Button>
                    </Flex>
                </Stack>
            )}
            <SoloReplacementConfirmation
                open={joining.confirmation !== null}
                onConfirm={joining.confirm}
                onDismiss={joining.dismiss}
            />
        </div>
    )
}
