// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
    // The outcome the store already held when this modal opened. It belongs to an
    // earlier open, so it is neither shown nor acted on.
    const [priorOutcome, setPriorOutcome] = useState<object | null>(null)
    // Bumped when this open ends, so a save started earlier cannot close a later open.
    const openInstance = useRef(0)
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
        if (open) {
            setPriorOutcome((peer ? peerChange : nameChange) ?? null)
            setDraft(
                target.kind === 'peer'
                    ? (target.node.nickname ?? target.node.name)
                    : (target.customName ?? target.shownName)
            )
        }
    }

    const endOpen = useCallback(() => {
        openInstance.current++
    }, [])
    useEffect(() => (open ? endOpen : undefined), [open, endOpen])

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
    // A save still running from an earlier open keeps the form busy (the store
    // refuses a second one); any other earlier outcome is not shown.
    const saving = change?.phase === 'saving'
    const outcome = change && change !== priorOutcome ? change : null
    const denied = peerChange !== priorOutcome && peerChange?.denied === true
    const canSaveName = !peer || (managementState === 'allowed' && !denied)
    const noGrant =
        peer !== null &&
        !canSaveName &&
        (denied || (managementState !== undefined && NO_GRANT_STATES.includes(managementState)))
    const shownName = target.kind === 'peer' ? target.node.displayName : target.shownName
    const hasDefaultToRestore = target.kind === 'peer' || target.customName !== null

    function currentOutcome() {
        const store = useClusterInvitationsStore.getState()
        return peer ? store.peerNameChanges[nodeUuid] : store.nameChange
    }

    /** Send one request and close only if it saved while this same open is still showing. */
    async function submit(send: () => Promise<void>): Promise<void> {
        if (currentOutcome()?.phase === 'saving') return
        const instance = openInstance.current
        await send()
        if (instance !== openInstance.current || currentOutcome()?.phase !== 'saved') return
        const store = useClusterInvitationsStore.getState()
        // Only this target's outcome: another target's in-flight save keeps its guard.
        if (peer) store.clearPeerNameChange(nodeUuid)
        else store.clearNameChange()
        onClose()
    }

    function saveName(name: string | null): Promise<void> {
        const store = useClusterInvitationsStore.getState()
        return submit(() => (peer ? store.setPeerName(nodeUuid, name) : store.setSelfName(name)))
    }

    function saveNickname(nickname: string | null): Promise<void> {
        return submit(() =>
            useClusterInvitationsStore.getState().setPeerNickname(nodeUuid, nickname)
        )
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
                        {outcome?.phase === 'error' && (
                            <Text kind="body/regular/sm" role="alert">
                                {outcome.message}
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
