// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from 'react'
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { MODULAR_CLUSTER_MANAGER_PORT } from '@/shared/constants/modular-runtime'
import getErrorString from '@/shared/utils/get-error-string'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useInvitablePeers } from '@/ui/hooks/useInvitablePeers'
import ActiveNode from './ActiveNode'
import AvailableNode from './AvailableNode'
import PendingInviteCard from './PendingInviteCard'
import JoinRequestCard from './JoinRequestCard'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'

export default function ClusterSettings() {
    const { connected, selfId, clusterId } = useConnectionStore()
    const members = useClusterInvitationsStore(state => state.members)
    const pendingInvites = useClusterInvitationsStore(state => state.pendingInvites)
    const joinSnapshot = useClusterInvitationsStore(state => state.joinSnapshot)
    const joinError = useClusterInvitationsStore(state => state.joinError)
    const selfNode = useNodesStore(state => (selfId ? (state.nodes.get(selfId) ?? null) : null))
    const [error, setError] = useState<string | null>(null)

    // `selfId` is the node UUID, so members correlate on `nodeUuid` (the
    // hostname `m.id` is display only).
    const connectedMembers = useMemo(
        () => members.filter(m => m.nodeUuid !== selfId && m.state === 'member'),
        [members, selfId]
    )

    // The backend exposes no cluster-level "created date"; the closest durable
    // signal is this node's own membership timestamp (set when it created or
    // joined the cluster). Read it from the self entry in the roster.
    const joinedAtLabel = useMemo(() => {
        const self = selfId ? members.find(m => m.nodeUuid === selfId) : null
        if (!self?.joinedAt) return null
        return new Date(self.joinedAt).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        })
    }, [members, selfId])

    const pendingMembers = useMemo(
        () => members.filter(m => m.nodeUuid !== selfId && m.state !== 'member'),
        [members, selfId]
    )

    const availableNodes = useInvitablePeers()

    const handleLeaveCluster = useCallback(async () => {
        if (!selfId || !clusterId) return
        try {
            const result = await window.pairApi.nodes.removeMember(selfId)
            if (!result.removed) {
                setError('Failed to leave the cluster. Please try again.')
                return
            }
            setError(null)
        } catch (err) {
            setError(getErrorString(err))
        }
    }, [selfId, clusterId])

    return (
        <Stack gap="6" className="relative py-8 px-3">
            {error && <InlineErrorBanner severity="error" message={error} />}
            {joinError && <InlineErrorBanner severity="error" message={joinError} />}

            <Flex wrap="wrap" align="start" justify="start" gap="4">
                <div className="settings-card pair-paper p-4">
                    <Flex justify="between" gap="4" className="mt-1">
                        <Stack gap="1">
                            <Text kind="body/semibold/md">IP</Text>
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                {selfNode?.ipAddress ?? ''}
                            </Text>
                        </Stack>
                        <Stack gap="1">
                            <Text kind="body/semibold/md">Pairing port</Text>
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                {MODULAR_CLUSTER_MANAGER_PORT}
                            </Text>
                        </Stack>
                    </Flex>
                </div>

                {clusterId && (
                    <div className="settings-card pair-paper p-4">
                        <Flex justify="between" align="center" gap="2">
                            <Stack gap="1">
                                <Text kind="body/semibold/md">Joined</Text>
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    {joinedAtLabel ?? '—'}
                                </Text>
                            </Stack>
                            <Button
                                kind="tertiary"
                                color="danger"
                                size="small"
                                disabled={!connected || !selfId}
                                onClick={handleLeaveCluster}
                            >
                                Leave
                            </Button>
                        </Flex>
                    </div>
                )}
            </Flex>

            <Stack gap="2">
                <Text kind="body/semibold/md" className="px-1">
                    Join requests
                </Text>
                {!joinSnapshot && (
                    <Text kind="body/regular/sm">
                        Join request status is unavailable. Reconnect to read it.
                    </Text>
                )}
                {joinSnapshot && !joinSnapshot.supported && (
                    <Text kind="body/regular/sm">Update PAIR to ask to join a cluster.</Text>
                )}
                {joinSnapshot?.supported && joinSnapshot.requests.length === 0 && (
                    <Text kind="body/regular/sm">No join requests.</Text>
                )}
                {joinSnapshot?.requests.map(request => (
                    <JoinRequestCard key={request.requestId} request={request} />
                ))}
            </Stack>

            {pendingInvites.length > 0 && (
                <Stack gap="2">
                    <Text kind="body/semibold/md" className="px-1">
                        Pending invites
                    </Text>

                    {pendingInvites.map(invite => (
                        <PendingInviteCard invite={invite} key={invite.inviteId} />
                    ))}
                </Stack>
            )}

            {connectedMembers.length > 0 && (
                <Stack gap="2">
                    <Text kind="body/semibold/md" className="px-1">
                        Connected nodes
                    </Text>

                    {connectedMembers.map(node => (
                        <ActiveNode node={node} key={node.nodeUuid} />
                    ))}
                </Stack>
            )}

            {pendingMembers.length > 0 && (
                <Stack gap="2">
                    <Text kind="body/semibold/md" className="px-1">
                        Pending
                    </Text>

                    {pendingMembers.map(node => (
                        <ActiveNode node={node} key={node.nodeUuid} />
                    ))}
                </Stack>
            )}

            {availableNodes.length > 0 && (
                <Stack gap="2">
                    <Text kind="body/semibold/md" className="px-1">
                        Available nodes to add
                    </Text>

                    {availableNodes.map(node => (
                        <AvailableNode node={node} key={node.id} />
                    ))}
                </Stack>
            )}
        </Stack>
    )
}
