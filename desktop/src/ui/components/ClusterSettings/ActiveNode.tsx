// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
    Button,
    Dropdown,
    Flex,
    Stack,
    Text,
    type DropdownEntry
} from '@nvidia/foundations-react-core'
import { MoreHoriz } from '@/ui/components/icons'
import type { ClusterNode } from '@/shared/types/cluster'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { RemoteManagementChoice } from '@/ui/components/InvitePairingPanel'

export default function ActiveNode({ node }: { node: ClusterNode }) {
    const focusNodeEngineSettings = useOverviewUiStore(state => state.focusNodeEngineSettings)
    const connected = useConnectionStore(state => state.connected)
    const permissions = useClusterInvitationsStore(state => state.remoteManagement)
    const permissionError = useClusterInvitationsStore(state => state.remoteManagementError)
    const permissionLoading = useClusterInvitationsStore(state => state.remoteManagementLoading)
    const change = useClusterInvitationsStore(state => state.remoteManagementChanges[node.nodeUuid])
    const [draft, setDraft] = useState<boolean | null>(null)
    const draftGeneration = useRef(0)
    const invalidateDraft = useCallback(() => {
        draftGeneration.current++
    }, [])
    useEffect(() => {
        invalidateDraft()
        setDraft(null)
        return invalidateDraft
    }, [node.nodeUuid, invalidateDraft])
    const grant = permissions?.grants.find(value => value.peerUuid === node.nodeUuid)
    const allowed = draft ?? grant?.allowed ?? false
    const saving = change?.phase === 'saving'
    const editable =
        connected &&
        !permissionLoading &&
        !!permissions?.revision &&
        !!grant &&
        grant.state !== 'unavailable'
    function choosePermission(allowed: boolean): void {
        draftGeneration.current++
        setDraft(allowed)
    }
    async function savePermission(): Promise<void> {
        const generation = ++draftGeneration.current
        await useClusterInvitationsStore.getState().setRemoteManagement(node.nodeUuid, allowed)
        if (
            generation === draftGeneration.current &&
            useClusterInvitationsStore.getState().remoteManagementChanges[node.nodeUuid]?.phase ===
                'saved'
        )
            setDraft(null)
    }
    // Prefer the live, address-ranked discovery IP (same source the node cards
    // use) over the cluster-manager roster IP, which is a pairing-time snapshot
    // (self is hardcoded 127.0.0.1, peers never re-ranked). Fall back to the
    // roster IP for a member discovery has not seen (offline / not yet found).
    const discoveredIp = useNodesStore(state => state.nodes.get(node.nodeUuid)?.ipAddress)
    const ipAddress = discoveredIp ?? node.ipAddress

    // Every node key is the stable UUID (`nodeUuid`) — the nodes-store key, the
    // engine-settings focus id, and the removal target. `node.id` is the display
    // hostname only.
    const items: DropdownEntry[] = useMemo(
        () => [
            {
                children: 'Edit node',
                onSelect: () => focusNodeEngineSettings(node.nodeUuid)
            },
            {
                children: 'Remove from cluster',
                danger: true,
                onSelect: () => {
                    void window.pairApi.nodes.removeMember(node.nodeUuid)
                }
            }
        ],
        [node.nodeUuid, focusNodeEngineSettings]
    )

    return (
        <div className="settings-card node-item-card pair-paper p-4">
            <Flex align="center" justify="between" gap="2">
                <Stack gap="1">
                    <Flex align="center" gap="1">
                        <Text kind="body/bold/sm" className="uppercase">
                            {node.name}
                        </Text>
                    </Flex>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {ipAddress}
                    </Text>
                </Stack>

                <Dropdown
                    items={items}
                    showChevron={false}
                    size="small"
                    aria-label={`Open ${node.name} menu`}
                >
                    <MoreHoriz style={{ fontSize: 16 }} />
                </Dropdown>
            </Flex>
            {node.state === 'member' && (
                <Stack gap="2" className="pt-3">
                    <Text kind="body/regular/sm">
                        Choose whether {node.name} may manage this device.
                    </Text>
                    <RemoteManagementChoice
                        allowed={allowed}
                        onChange={choosePermission}
                        disabled={!editable || saving}
                    />
                    <Text kind="body/regular/sm" role="status">
                        {grant?.state === 'allowed'
                            ? 'Saved: allowed to manage this device.'
                            : grant?.state === 'denied'
                              ? 'Saved: blocked from changing this device.'
                              : grant?.state === 'unreviewed'
                                ? 'Not reviewed: this peer cannot manage this device.'
                                : permissionError ||
                                  'Reading the saved permission for this device.'}
                    </Text>
                    {change && change.phase !== 'saved' && change.phase !== 'saving' && (
                        <Text kind="body/regular/sm" role="alert">
                            {change.phase === 'conflict'
                                ? 'The saved permission changed. Your choice is kept; review the saved result before saving again.'
                                : 'The save could not be confirmed. Your choice is kept; refresh the saved permission before trying again.'}
                            {change.message ? ` ${change.message}` : ''}
                        </Text>
                    )}
                    <Flex gap="2" justify="end">
                        <Button
                            kind="secondary"
                            size="small"
                            disabled={!connected || saving}
                            onClick={() =>
                                void useClusterInvitationsStore.getState().refreshRemoteManagement()
                            }
                        >
                            Refresh permission
                        </Button>
                        <Button
                            kind="primary"
                            color="brand"
                            size="small"
                            disabled={
                                !editable ||
                                saving ||
                                (draft === null && grant?.state !== 'unreviewed')
                            }
                            aria-busy={saving}
                            onClick={() => void savePermission()}
                        >
                            Save permission
                        </Button>
                    </Flex>
                </Stack>
            )}
        </div>
    )
}
