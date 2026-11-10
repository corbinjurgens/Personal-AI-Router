// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    ModalContent,
    ModalDialog,
    ModalRoot,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { ClusterNode } from '@/shared/types/cluster'
import { DialogHeader } from '@/ui/components/DialogHeader'
import { useBlurOnOpen } from '@/ui/hooks/useBlurOnOpen'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import {
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { duplicateShownName } from '@/ui/utils/node-name-labels'

/** What the modal renames: this node, or another cluster member. */
export type NodeNameTarget =
    | {
          kind: 'self'
          nodeUuid: string
          /** The node's shown name. */
          shownName: string
          /** The node's user-set name; `null` while the generated default is shown. */
          customName: string | null
      }
    | { kind: 'peer'; node: ClusterNode }

interface NodeNameModalProps {
    target: NodeNameTarget
    open: boolean
    onClose: () => void
}

/** Management states in which the target has told this node it may not rename it. */
const NO_GRANT_STATES = ['denied', 'unreviewed', 'unavailable', 'unsupported']

/**
 * Rename this node, or a peer. This node's name is saved through its own
 * cluster manager. A peer can be renamed on that peer only while it has allowed
 * this node remote management (the peer enforces it); a nickname is always
 * available and only changes what this node shows.
 */
export function NodeNameModal({ target, open, onClose }: NodeNameModalProps) {
    useBlurOnOpen(open)
    const [draft, setDraft] = useState('')
    const [prefilledOpen, setPrefilledOpen] = useState(false)
    const peer = target.kind === 'peer' ? target.node : null
    const nodeUuid = target.kind === 'peer' ? target.node.nodeUuid : target.nodeUuid
    const nameChange = useClusterInvitationsStore(state => state.nameChange)
    const peerChange = useClusterInvitationsStore(state => state.peerNameChanges[nodeUuid])
    const members = useClusterInvitationsStore(useShallow(state => state.members))
    const managementState = useNodeConfigurationStore(state =>
        peer ? state.management.get(nodeUuid)?.state : undefined
    )
    const overviewNodes = useOverviewNodes()

    // Prefill when the modal opens, not on every later name update.
    if (open !== prefilledOpen) {
        setPrefilledOpen(open)
        if (open)
            setDraft(
                target.kind === 'peer'
                    ? (target.node.nickname ?? target.node.name)
                    : (target.customName ?? target.shownName)
            )
    }

    useEffect(() => {
        if (!open) return
        const store = useClusterInvitationsStore.getState()
        store.clearNameChange()
        store.clearPeerNameChange(nodeUuid)
    }, [open, nodeUuid])

    // The shared permission read for this peer; Save is offered only on `allowed`.
    const watchedPeerUuid = open && peer ? nodeUuid : null
    useEffect(
        () => (watchedPeerUuid ? watchNodeManagement(watchedPeerUuid, watchedPeerUuid) : undefined),
        [watchedPeerUuid]
    )

    const otherNames = useMemo(
        () => [
            ...overviewNodes.filter(node => node.id !== nodeUuid).map(node => node.name),
            ...members
                .filter(member => member.nodeUuid !== nodeUuid)
                .map(member => member.displayName)
        ],
        [overviewNodes, members, nodeUuid]
    )
    const trimmed = draft.trim()
    const duplicate = duplicateShownName(trimmed, otherNames)
    const change = peer ? peerChange : nameChange
    const saving = change?.phase === 'saving'
    const canSaveName = !peer || (managementState === 'allowed' && !peerChange?.denied)
    const noGrant =
        peer !== null &&
        !canSaveName &&
        (peerChange?.denied === true ||
            (managementState !== undefined && NO_GRANT_STATES.includes(managementState)))
    const shownName = target.kind === 'peer' ? target.node.displayName : target.shownName
    const hasDefaultToRestore = target.kind === 'peer' || target.customName !== null

    function closeWhenSaved(): void {
        const store = useClusterInvitationsStore.getState()
        const outcome = peer ? store.peerNameChanges[nodeUuid] : store.nameChange
        if (outcome?.phase !== 'saved') return
        store.clearNameChange()
        store.clearPeerNameChange(nodeUuid)
        onClose()
    }

    async function saveName(name: string | null): Promise<void> {
        const store = useClusterInvitationsStore.getState()
        if (peer) await store.setPeerName(nodeUuid, name)
        else await store.setSelfName(name)
        closeWhenSaved()
    }

    async function saveNickname(nickname: string | null): Promise<void> {
        await useClusterInvitationsStore.getState().setPeerNickname(nodeUuid, nickname)
        closeWhenSaved()
    }

    return (
        <ModalRoot open={open} onOpenChange={next => !next && onClose()} hideCloseButton>
            <ModalDialog>
                <ModalContent className="no-drag-elements max-content-modal">
                    <DialogHeader onClose={onClose}>Rename node</DialogHeader>
                    <Stack gap="4" className="pt-2">
                        <FormField slotLabel="Node name">
                            <TextInput
                                value={draft}
                                aria-label="Node name"
                                onValueChange={setDraft}
                                onKeyDown={event => {
                                    if (event.key !== 'Enter' || !trimmed || saving) return
                                    if (canSaveName) void saveName(trimmed)
                                    else void saveNickname(trimmed)
                                }}
                                disabled={saving}
                            />
                        </FormField>
                        {peer && canSaveName && (
                            <Text kind="body/regular/sm">
                                {`Renames ${shownName} on that device. Every node in the cluster will show the new name.`}
                            </Text>
                        )}
                        {noGrant && (
                            <Text kind="body/regular/sm">
                                {`${shownName} has not allowed this device to manage it, so you can't rename it from here. You can still give it a nickname.`}
                            </Text>
                        )}
                        {peer && (
                            <Text kind="body/regular/sm">
                                A nickname is only used on this device. Other devices keep showing
                                the node&apos;s own name.
                            </Text>
                        )}
                        {duplicate !== null && (
                            <Text kind="body/regular/sm" role="status">
                                {`Another node is already shown as "${duplicate}". You can still save; PAIR adds a hardware label to tell them apart.`}
                            </Text>
                        )}
                        {change?.phase === 'error' && (
                            <Text kind="body/regular/sm" role="alert">
                                {change.message}
                            </Text>
                        )}
                        <Flex gap="2" justify="end" className="flex-wrap">
                            {peer?.nickname !== undefined && (
                                <Button
                                    kind="secondary"
                                    disabled={saving}
                                    onClick={() => void saveNickname(null)}
                                >
                                    Clear nickname
                                </Button>
                            )}
                            {canSaveName && hasDefaultToRestore && (
                                <Button
                                    kind="secondary"
                                    disabled={saving}
                                    onClick={() => void saveName(null)}
                                >
                                    Use default name
                                </Button>
                            )}
                            {peer && (
                                <Button
                                    kind={canSaveName ? 'secondary' : 'primary'}
                                    color={canSaveName ? undefined : 'brand'}
                                    disabled={saving || !trimmed}
                                    onClick={() => void saveNickname(trimmed)}
                                >
                                    Set nickname
                                </Button>
                            )}
                            {canSaveName && (
                                <Button
                                    kind="primary"
                                    color="brand"
                                    disabled={saving || !trimmed}
                                    aria-busy={saving}
                                    onClick={() => void saveName(trimmed)}
                                >
                                    Save
                                </Button>
                            )}
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
