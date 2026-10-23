// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from 'react'
import type { AskToJoinRequest } from '@/shared/types/cluster'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

export function useAskToJoin() {
    const snapshot = useClusterInvitationsStore(state => state.joinSnapshot)
    const submitting = useClusterInvitationsStore(state => state.joinBusy.outgoing === true)
    const error = useClusterInvitationsStore(state => state.joinError)
    const [confirmation, setConfirmation] = useState<AskToJoinRequest | null>(null)
    const dismiss = useCallback(() => setConfirmation(null), [])
    const blocked = !snapshot
        ? 'Read the service status before asking to join.'
        : !snapshot.supported
          ? 'Update PAIR to ask to join a cluster.'
          : snapshot.blockedReason === 'already-clustered'
            ? 'This device already has cluster peers. Leave its current cluster before asking to join another.'
            : !snapshot.canAskToJoin
              ? snapshot.blockedReason === 'pairing-busy'
                  ? 'Finish the current pairing or join request before asking to join another cluster.'
                  : 'The service cannot confirm this device is eligible. Refresh its status before asking to join.'
              : null
    const start = (address: string, port?: number, nodeId?: string) => {
        if (blocked || submitting || !address.trim()) return
        const request = { address: address.trim(), port, nodeId, replaceSoloCluster: false }
        if (snapshot?.requiresSoloReplacement) setConfirmation(request)
        else void useClusterInvitationsStore.getState().askToJoin(request)
    }
    const confirm = () => {
        if (!confirmation) return
        const request = confirmation
        setConfirmation(null)
        if (!blocked)
            void useClusterInvitationsStore.getState().askToJoin({
                ...request,
                replaceSoloCluster: snapshot?.requiresSoloReplacement === true
            })
    }
    return { start, blocked, submitting, error, confirmation, confirm, dismiss }
}
