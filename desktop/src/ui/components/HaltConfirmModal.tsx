// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { HALT_CONSEQUENCE } from '@/ui/constants/availability-copy'

/** The one confirmation shown before a local Halt, from the header and from the node card. */
export function HaltConfirmModal({
    open,
    onOpenChange,
    onConfirm
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    onConfirm: () => void
}) {
    return (
        <ConfirmModal
            open={open}
            onOpenChange={onOpenChange}
            title="Halt this PC?"
            message={HALT_CONSEQUENCE}
            confirmLabel="Halt"
            confirmColor="danger"
            onConfirm={onConfirm}
        />
    )
}
