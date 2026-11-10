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
import { DialogHeader } from '@/ui/components/DialogHeader'
import { useBlurOnOpen } from '@/ui/hooks/useBlurOnOpen'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { duplicateShownName } from '@/ui/utils/node-name-labels'

interface NodeNameModalProps {
    /** UUID of the node being renamed. */
    nodeUuid: string
    /** The node's shown name. */
    shownName: string
    /** The node's user-set name; `null` while the generated default is shown. */
    customName: string | null
    open: boolean
    onClose: () => void
}

/** Rename this node; the cluster manager validates and stores the name. */
export function NodeNameModal({
    nodeUuid,
    shownName,
    customName,
    open,
    onClose
}: NodeNameModalProps) {
    useBlurOnOpen(open)
    const [draft, setDraft] = useState('')
    const [prefilledOpen, setPrefilledOpen] = useState(false)
    const nameChange = useClusterInvitationsStore(state => state.nameChange)
    const members = useClusterInvitationsStore(useShallow(state => state.members))
    const overviewNodes = useOverviewNodes()

    // Prefill when the modal opens, not on every later name update.
    if (open !== prefilledOpen) {
        setPrefilledOpen(open)
        if (open) setDraft(customName ?? shownName)
    }

    useEffect(() => {
        if (open) useClusterInvitationsStore.getState().clearNameChange()
    }, [open])

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
    const saving = nameChange?.phase === 'saving'

    async function save(name: string | null): Promise<void> {
        const store = useClusterInvitationsStore.getState()
        await store.setSelfName(name)
        if (useClusterInvitationsStore.getState().nameChange?.phase !== 'saved') return
        store.clearNameChange()
        onClose()
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
                                    if (event.key === 'Enter' && trimmed && !saving)
                                        void save(trimmed)
                                }}
                                disabled={saving}
                            />
                        </FormField>
                        {duplicate !== null && (
                            <Text kind="body/regular/sm" role="status">
                                {`Another node is already shown as "${duplicate}". You can still save; PAIR adds a hardware label to tell them apart.`}
                            </Text>
                        )}
                        {nameChange?.phase === 'error' && (
                            <Text kind="body/regular/sm" role="alert">
                                {nameChange.message}
                            </Text>
                        )}
                        <Flex gap="2" justify="end">
                            {customName !== null && (
                                <Button
                                    kind="secondary"
                                    disabled={saving}
                                    onClick={() => void save(null)}
                                >
                                    Use default name
                                </Button>
                            )}
                            <Button
                                kind="primary"
                                color="brand"
                                disabled={saving || !trimmed}
                                aria-busy={saving}
                                onClick={() => void save(trimmed)}
                            >
                                Save
                            </Button>
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
