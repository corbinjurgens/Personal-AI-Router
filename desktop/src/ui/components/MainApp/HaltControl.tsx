// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import { HALT_CONSEQUENCE } from '@/ui/constants/availability-copy'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    useNodeConfigurationStore,
    watchNodeAvailability
} from '@/ui/stores/node-configuration.store'

export default function HaltControl({ targetId }: { targetId: string }) {
    const selfId = useConnectionStore(state => state.selfId)
    const connected = useConnectionStore(state => state.connected)
    const entry = useNodeConfigurationStore(state => state.availability.get(targetId))
    const [confirmOpen, setConfirmOpen] = useState(false)
    const local = targetId === selfId
    useEffect(() => {
        if (!local) return
        return watchNodeAvailability(targetId, undefined, true)
    }, [local, targetId])
    if (!local) return null

    const busy = entry?.pending !== undefined
    const disabled = !connected || !entry?.current || entry.stale || !entry.value || busy
    const resume = entry?.value === 'paused' || entry?.value === 'draining'
    const request = () => {
        if (disabled) return
        void useNodeConfigurationStore
            .getState()
            .requestAvailability(
                targetId,
                resume ? 'available' : 'paused',
                resume ? undefined : 'immediate'
            )
    }

    return (
        <Stack gap="2">
            <Button
                kind="primary"
                color={resume ? 'brand' : 'danger'}
                size="medium"
                disabled={disabled}
                aria-busy={busy}
                onClick={() => (resume ? request() : setConfirmOpen(true))}
            >
                {resume ? 'Resume' : 'Halt'}
            </Button>
            {resume && entry?.pausedBy === 'peer' && (
                <Text kind="body/regular/sm" role="status">
                    Paused by another node
                </Text>
            )}
            {entry?.error && (
                <Text kind="body/regular/sm" role="alert">
                    {entry.error}
                </Text>
            )}
            {entry?.readError && (
                <Text kind="body/regular/sm" role="alert">
                    {entry.readError}
                </Text>
            )}
            <ConfirmModal
                open={confirmOpen}
                onOpenChange={setConfirmOpen}
                title="Halt this PC?"
                message={HALT_CONSEQUENCE}
                confirmLabel="Halt"
                confirmColor="danger"
                onConfirm={() => {
                    if (!resume) request()
                }}
            />
        </Stack>
    )
}
